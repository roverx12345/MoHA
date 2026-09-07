"""Allocation uses the pinned renderer and processor estimates, without inference."""
import unittest
from types import SimpleNamespace
from dataclasses import replace
from moha.execution import ExecutionPolicy, allocate, plan_signature
from test_runtime import VIDEO_OS_AVAILABLE


def media(width=640, height=360, fps=24):
    return SimpleNamespace(width=width, height=height, frame_rate=fps,
        duration_seconds=60, source_sha256="unit-source")


def renderer(**limits):
    from video_os.media.renderer import ViewportRenderer, RendererConfig
    from video_os.media.sensory import Qwen3OmniMediaTokenProfile, AdaptivePackingPolicy
    from video_os.providers.core import default_perception_budget
    result = ViewportRenderer.__new__(ViewportRenderer)
    result.budget = replace(default_perception_budget(), **{"f_view":128, "p_view":262144,
        "p_call":33554432, "b_video":16384, "c_sensor_max":None, **limits})
    result.config = RendererConfig()
    result.packing_policy = AdaptivePackingPolicy()
    result.token_profile = Qwen3OmniMediaTokenProfile()
    return result


@unittest.skipUnless(VIDEO_OS_AVAILABLE, "requires the pinned runtime")
class AllocationTests(unittest.TestCase):
    def test_source_ratios_are_edge_ratios_without_upscaling(self):
        for width, height in [(640, 360), (1920, 1080), (360, 640)]:
            r = renderer(p_view=4000000, p_call=128*4000000, b_video=100000)
            plans = [allocate(r, media(width, height), (0, 1), ExecutionPolicy(source_scale=s))
                     for s in (.5, .75, 1)]
            self.assertEqual([p['resolution'] for p in plans],
                             [[int(width*s)//2*2, int(height*s)//2*2] for s in (.5,.75,1)])

    def test_modes_converge_when_budget_is_not_binding(self):
        plans = [allocate(renderer(), media(), (5, 15), ExecutionPolicy(priority=p))
                 for p in ('temporal', 'spatial', 'balanced')]
        self.assertEqual(len({plan_signature(p) for p in plans}), 1)
        self.assertEqual(plans[0]['resolution'], [640, 360])
        self.assertEqual(plans[0]['frames'], 10)

    def test_pressure_trades_time_and_space_under_identical_limits(self):
        r = renderer(p_view=2073600, p_call=2073600*128, b_video=5000)
        source = media(1920,1080)
        plans = {mode: allocate(r, source, (0,60), ExecutionPolicy(frames=128, priority=mode))
                 for mode in ('temporal','spatial','balanced')}
        t,s,b = [plans[k] for k in ('temporal','spatial','balanced')]
        self.assertGreater(t['frames'], s['frames'])
        self.assertLess(t['source_scale_realized'], s['source_scale_realized'])
        self.assertLessEqual(s['frames'], b['frames'])
        self.assertLessEqual(b['frames'], t['frames'])
        for plan in plans.values():
            self.assertLessEqual(plan['video_tokens'], r.budget.b_video)
            self.assertEqual(plan['window'], [0,60])
            self.assertLessEqual(plan['frames']*plan['resolution'][0]*plan['resolution'][1], r.budget.p_call)
            # The old renderer must accept the selected packet without changing it.
            count, dims, _ = r._adaptive_window_shape(metadata=source, crop=(0,0,1920,1080),
                duration=60, sampling_policy='uniform', include_audio=False,
                **plan['experiment_render'])
            self.assertEqual((count,list(dims)), (plan['frames'],plan['resolution']))

    def test_source_frames_cap_target_and_qwen_minimum_is_accounted(self):
        source = media(fps=2)
        self.assertEqual(allocate(renderer(), source, (0,.5), ExecutionPolicy(frames=128))['frames'],1)
        plans = [allocate(renderer(), media(), (0,10), ExecutionPolicy(source_scale=s)) for s in (.5,.75)]
        self.assertNotEqual(plans[0]['resolution'],plans[1]['resolution'])
        # Both can enter the processor's minimum-pixel regime, despite different source detail.
        for plan in plans:
            profile=renderer().token_profile
            self.assertEqual(plan['video_tokens'],profile.estimate_adaptive_video(10,
                frames=plan['frames'],pixels_per_frame=plan['resolution'][0]*plan['resolution'][1],
                include_audio=False,frame_width=plan['resolution'][0],frame_height=plan['resolution'][1]))

    def test_unfit_minimum_does_not_overrun_budget(self):
        from video_os.core.errors import BudgetExceeded
        with self.assertRaises(BudgetExceeded):
            allocate(renderer(b_video=1),media(),(0,10),ExecutionPolicy(frames=128))


if __name__ == '__main__':
    unittest.main()
