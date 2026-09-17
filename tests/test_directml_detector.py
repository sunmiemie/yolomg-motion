import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yolomg_motion import directml_detector as directml


def test_directml_missing_provider_has_actionable_error(monkeypatch):
    modules = {
        "onnx": SimpleNamespace(),
        "onnxruntime": SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"]),
    }
    monkeypatch.setattr(directml.importlib, "import_module", modules.__getitem__)
    with pytest.raises(RuntimeError, match="DmlExecutionProvider is unavailable"):
        directml._dependencies()


def test_missing_export_dependency_does_not_download(monkeypatch):
    def missing(name):
        raise ImportError(name)
    monkeypatch.setattr(directml.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="No model was downloaded"):
        directml._dependencies()


@pytest.mark.parametrize("providers,accepted", [
    (["DmlExecutionProvider", "CPUExecutionProvider"], True),
    (["CPUExecutionProvider"], False),
    ([], False),
])
def test_actual_profile_requires_gpu_execution(tmp_path, providers, accepted):
    profile = tmp_path / "profile.json"
    events = [
        {"cat": "Node", "name": f"node_{i}_kernel_time", "args": {"provider": provider}}
        for i, provider in enumerate(providers)
    ]
    # A provider mentioned in a fence event is not evidence of a GPU kernel.
    events.append({"cat": "Node", "name": "node_fence_before", "args": {"provider": "DmlExecutionProvider"}})
    profile.write_text(json.dumps(events), encoding="utf-8")
    if accepted:
        assert directml._executed_provider_counts(profile) == {
            "DmlExecutionProvider": 1, "CPUExecutionProvider": 1,
        }
    else:
        with pytest.raises(RuntimeError, match="did not execute any profiled"):
            directml._executed_provider_counts(profile)


@pytest.mark.parametrize("gpu_present", [True, False])
def test_session_verifies_provider_and_removes_temporary_profile(tmp_path, gpu_present):
    created = []

    class Session:
        def __init__(self, path, sess_options, providers):
            self.options = sess_options
            self.providers = providers
            self.fallback_disabled = False
            self.ran = False
            self.profile = Path(sess_options.profile_file_prefix + ".json")
            created.append(self)

        def disable_fallback(self):
            self.fallback_disabled = True

        def get_providers(self):
            return ["DmlExecutionProvider", "CPUExecutionProvider"] if gpu_present else ["CPUExecutionProvider"]

        def run(self, names, inputs):
            assert self.fallback_disabled
            self.ran = True
            assert names == ["rawprediction"]
            return [np.zeros((1, 10, 6), dtype=np.float32)]

        def end_profiling(self):
            self.profile.write_text(json.dumps([
                {"cat": "Node", "name": "dml_kernel_time", "args": {"provider": "DmlExecutionProvider"}},
                {"cat": "Node", "name": "cpu_kernel_time", "args": {"provider": "CPUExecutionProvider"}},
            ]), encoding="utf-8")
            return str(self.profile)

    detector = directml.DirectMLYOLOMGDetector.__new__(directml.DirectMLYOLOMGDetector)
    detector._ort = SimpleNamespace(
        SessionOptions=SimpleNamespace,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        InferenceSession=Session,
    )
    detector._threads = 4
    detector.device_id = 0
    detector.backend_info = {}
    detector._onnx_path = lambda rgb, motion: tmp_path / "model.onnx"
    tensor = np.zeros((1, 3, 32, 64), dtype=np.float32)
    if gpu_present:
        assert detector._first_run(tensor, tensor, {}).shape == (1, 10, 6)
        assert detector.backend_info["execution_verified"]
        assert detector.backend_info["cpu_graph_nodes"] == 1
    else:
        with pytest.raises(RuntimeError, match="fell back"):
            detector._first_run(tensor, tensor, {})
        assert not created[0].ran
        assert not detector.backend_info["execution_verified"]
    session = created[0]
    assert session.fallback_disabled
    assert session.options.execution_mode == "sequential"
    assert not session.options.enable_mem_pattern
    assert session.options.intra_op_num_threads == 4
    assert not session.profile.parent.exists()
