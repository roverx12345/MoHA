"""No network, credentials, video files or third-party packages required."""
from .loop import Calibrator, SearchPolicy
from .models import Episode, Sample, ValidationPolicy
from .store import RunStore


class DemoRunner:
    def run(self, harness, sample, repeat):
        answer = "A" if harness.overview else "B"
        return Episode(sample.sample_id, sample.id, harness.id, repeat, list(sample.video_key),
                       sample.expected_answer, answer, "completed", usage={"video_tokens": 100})


class DemoJudge:
    def diagnose(self, payload):
        return {"status": "valid", "failure": "orientation", "reason": "synthetic example", "confidence": 1.0}


class DemoSelector:
    def select(self, diagnoses, harness, available, history):
        return {"status": "valid", "candidate_id": "planner.module.overview", "reason": "synthetic example"}


def sample(name):
    return Sample(name, name, "synthetic", name, {"question": "Synthetic question", "options": {"A": "yes", "B": "no"}}, "A")


def run_demo(output, resume=False):
    with RunStore(output, {"demo": "moha_demo_v1", "synthetic": True}, resume=resume) as store:
        return Calibrator(runner=DemoRunner(), judge=DemoJudge(), selector=DemoSelector(), store=store,
            calibration=[sample("cal")], validation=[sample(f"val{i}") for i in range(8)],
            search=SearchPolicy(max_rounds=2), validation_policy=ValidationPolicy(bootstrap_samples=100)).run()
