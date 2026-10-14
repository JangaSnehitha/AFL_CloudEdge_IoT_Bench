"""Unit tests for dynamic network conditions, outage intervals, and scenario configurations."""

import unittest

from src.client_profiles import create_client_profile
from src.scenarios import (
    DynamicNetworkManager,
    NetworkConditionEvent,
    OutageEvent,
    ScenarioConfig,
    get_named_scenario,
    get_scenario_s1,
    get_scenario_s2,
    get_scenario_s3,
    get_scenario_s4,
    get_scenario_s5,
)


class TestScenarios(unittest.TestCase):
    """Test suite for dynamic network schedules, outages, and scenario templates."""

    def test_scheduled_network_spikes_and_recovery_work(self) -> None:
        """Verify dynamic network manager accurately applies latency spike and recovers."""
        base_prof = create_client_profile(client_id=0, speed_tier="medium")
        base_lat = base_prof.base_latency  # 0.050s
        base_dl = base_prof.download_bandwidth  # 524288 B/s

        # Spike between t=2.0 and t=5.0
        spike_event = NetworkConditionEvent(
            event_type="latency_spike",
            start_time=2.0,
            end_time=5.0,
            client_ids=[0],
            latency_delta=0.100,  # +100ms
            bandwidth_multiplier=0.5,  # 50% bandwidth
        )

        net_mgr = DynamicNetworkManager(network_events=[spike_event], outage_events=[])

        # 1. Before spike (t=1.0)
        dl_1, ul_1, lat_1 = net_mgr.get_effective_network(0, 1.0, base_prof)
        self.assertAlmostEqual(lat_1, base_lat)
        self.assertAlmostEqual(dl_1, base_dl)

        # 2. During spike (t=3.0)
        dl_2, ul_2, lat_2 = net_mgr.get_effective_network(0, 3.0, base_prof)
        self.assertAlmostEqual(lat_2, base_lat + 0.100)
        self.assertAlmostEqual(dl_2, base_dl * 0.5)

        # 3. After recovery (t=6.0)
        dl_3, ul_3, lat_3 = net_mgr.get_effective_network(0, 6.0, base_prof)
        self.assertAlmostEqual(lat_3, base_lat)
        self.assertAlmostEqual(dl_3, base_dl)

    def test_outage_detection_and_interval_intersection(self) -> None:
        """Verify client outages are detected accurately over virtual-time intervals."""
        outage = OutageEvent(
            client_ids=[1],
            start_time=2.0,
            end_time=4.0,
            failure_stage="BEFORE_UPLOAD",
        )
        net_mgr = DynamicNetworkManager(network_events=[], outage_events=[outage])

        # Client 1 is in outage at t=3.0, but not at t=1.0 or t=5.0
        in_outage, _ = net_mgr.is_in_outage(1, 3.0)
        self.assertTrue(in_outage)

        in_outage_before, _ = net_mgr.is_in_outage(1, 1.0)
        self.assertFalse(in_outage_before)

        in_outage_after, _ = net_mgr.is_in_outage(1, 5.0)
        self.assertFalse(in_outage_after)

        # Client 0 is never in outage
        in_outage_c0, _ = net_mgr.is_in_outage(0, 3.0)
        self.assertFalse(in_outage_c0)

        # Intersecting interval detection
        inter = net_mgr.get_intersecting_outage(1, 1.5, 2.5)  # Overlaps [2.0, 4.0]
        self.assertIsNotNone(inter)

        non_inter = net_mgr.get_intersecting_outage(1, 0.0, 1.5)  # Before [2.0, 4.0]
        self.assertIsNone(non_inter)

    def test_named_scenarios_factory(self) -> None:
        """Verify all standard named scenarios S1 through S5 build valid configurations."""
        for name in ["S1", "S2", "S3", "S4", "S5"]:
            sc = get_named_scenario(name, n_clients=5, target_updates=15)
            self.assertEqual(sc.n_clients, 5)
            self.assertEqual(sc.target_applied_updates, 15)
            self.assertGreater(sc.max_virtual_time, 0.0)

        with self.assertRaises(ValueError):
            get_named_scenario("S99_invalid")


if __name__ == "__main__":
    unittest.main()
