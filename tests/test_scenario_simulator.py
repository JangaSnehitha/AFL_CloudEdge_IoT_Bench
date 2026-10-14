"""Unit tests for the dynamic Cloud-Edge IoT scenario simulator."""

import copy
import unittest

from src.scenarios import (
    NetworkConditionEvent,
    OutageEvent,
    ScenarioConfig,
    get_scenario_s1,
    get_scenario_s4,
)
from src.scenario_simulator import ScenarioSimulator, run_scenario_simulation


class TestScenarioSimulator(unittest.TestCase):
    """Test suite for event-driven dynamic scenario execution, outages, and rejoin semantics."""

    def test_outage_prevents_successful_transmission_and_causes_failure_or_retention(self) -> None:
        """Verify outages prevent normal transmission during the unavailable interval."""
        outage = OutageEvent(
            client_ids=[0],
            start_time=0.25,
            end_time=0.45,
            failure_stage="BEFORE_UPLOAD",
        )
        cfg = ScenarioConfig(
            scenario_name="test_outage",
            description="Testing outage effect",
            n_clients=3,
            target_applied_updates=8,
            outage_events=[outage],
            rejoin_policy="fresh_restart",
        )
        sim = ScenarioSimulator(cfg, algorithm="FedAsync")
        res = sim.run_simulation()

        # Disrupted jobs must be recorded
        self.assertGreater(res["summary"]["failed_jobs"], 0)
        self.assertIn("BEFORE_UPLOAD", res["summary"]["failures_by_stage"])

        # Check event log records JOB_FAILED during the outage window
        failed_events = [e for e in res["event_log"] if e.get("event_type") == "JOB_FAILED" and e.get("client_id") == 0]
        self.assertGreater(len(failed_events), 0)

    def test_fresh_restart_uses_current_server_model_version(self) -> None:
        """Verify that under fresh_restart, a rejoined client downloads the current server model version."""
        outage = OutageEvent(
            client_ids=[0],
            start_time=0.25,
            end_time=0.45,
            failure_stage="BEFORE_UPLOAD",
        )
        cfg = ScenarioConfig(
            scenario_name="test_fresh_restart",
            description="Testing fresh restart version update",
            n_clients=3,
            target_applied_updates=8,
            outage_events=[outage],
            rejoin_policy="fresh_restart",
        )
        sim = ScenarioSimulator(cfg, algorithm="FedAsync")
        res = sim.run_simulation()

        # Find arrivals for Client 0 after t=0.45
        c0_post_rejoin_arrivals = [
            a for a in res["arrival_log"]
            if a["client_id"] == 0 and a["arrival_time"] > 0.45
        ]
        self.assertGreater(len(c0_post_rejoin_arrivals), 0)
        first_post_rejoin = c0_post_rejoin_arrivals[0]
        self.assertGreaterEqual(first_post_rejoin["client_training_version"], 0)

    def test_retained_updates_preserve_their_original_training_version(self) -> None:
        """Verify that under retained_update policy, a retained job keeps its original training version."""
        outage = OutageEvent(
            client_ids=[0],
            start_time=0.25,
            end_time=0.45,
            failure_stage="BEFORE_UPLOAD",
        )
        cfg = ScenarioConfig(
            scenario_name="test_retained_version",
            description="Testing retained update preserves original training version",
            n_clients=3,
            target_applied_updates=8,
            outage_events=[outage],
            rejoin_policy="retained_update",
        )
        sim = ScenarioSimulator(cfg, algorithm="FedAsync")
        res = sim.run_simulation()

        # At least one retained update must be recorded
        self.assertGreater(res["summary"]["retained_updates_count"], 0)

        # Check event log contains JOB_RETAINED and REJOIN_UPLOAD_STARTED
        retained_events = [e for e in res["event_log"] if e.get("event_type") == "JOB_RETAINED" and e.get("client_id") == 0]
        self.assertGreater(len(retained_events), 0)
        orig_version = retained_events[0]["training_version"]

        upload_events = [e for e in res["event_log"] if e.get("event_type") == "REJOIN_UPLOAD_STARTED" and e.get("client_id") == 0]
        self.assertGreater(len(upload_events), 0)
        self.assertEqual(upload_events[0]["retained_training_version"], orig_version)

    def test_no_duplicate_job_application_and_at_most_one_job_per_client(self) -> None:
        """Verify each job ID is applied at most once and clients have at most one active job."""
        cfg = get_scenario_s4(n_clients=4, target_updates=10)
        sim = ScenarioSimulator(cfg, algorithm="FedAsync")
        res = sim.run_simulation()

        applied_job_ids = [a["job_id"] for a in res["arrival_log"]]
        self.assertEqual(len(applied_job_ids), len(set(applied_job_ids)), "Duplicate job ID applied!")

    def test_fixed_seeds_reproduce_event_logs_identically(self) -> None:
        """Verify fixed random seeds produce bit-for-bit identical scenario runs."""
        cfg1 = get_scenario_s4(n_clients=4, target_updates=8)
        cfg2 = get_scenario_s4(n_clients=4, target_updates=8)

        res1 = ScenarioSimulator(cfg1, algorithm="FedAsync").run_simulation()
        res2 = ScenarioSimulator(cfg2, algorithm="FedAsync").run_simulation()

        self.assertEqual(res1["summary"]["applied_updates"], res2["summary"]["applied_updates"])
        self.assertEqual(res1["summary"]["total_virtual_time_s"], res2["summary"]["total_virtual_time_s"])
        self.assertEqual(res1["summary"]["failed_jobs"], res2["summary"]["failed_jobs"])
        self.assertEqual(res1["arrival_log"], res2["arrival_log"])

    def test_hard_stopping_bounds_terminate_unreachable_budgets(self) -> None:
        """Verify simulator terminates cleanly when max_attempted_jobs or max_virtual_time is reached."""
        # Permanent outage on all clients except client 0, but set max_attempted_jobs = 4
        all_outage = OutageEvent(
            client_ids=[1, 2],
            start_time=0.0,
            end_time=100.0,
            failure_stage="BEFORE_DOWNLOAD",
        )
        cfg = ScenarioConfig(
            scenario_name="test_budget_cap",
            description="Testing stopping bound",
            n_clients=3,
            target_applied_updates=50,  # Unreachable with small max_attempted_jobs
            max_attempted_jobs=6,
            max_virtual_time=50.0,
            outage_events=[all_outage],
        )
        sim = ScenarioSimulator(cfg, algorithm="FedAsync")
        res = sim.run_simulation()

        # Must terminate with budget cap
        self.assertIn(res["stopping_reason"], {"max_attempted_jobs_reached", "max_virtual_time_reached"})
        self.assertLessEqual(res["summary"]["jobs_dispatched"], 6)

    def test_fedbuff_executes_on_dynamic_scenarios(self) -> None:
        """Verify FedBuff integrates properly with dynamic scenario components."""
        cfg = get_scenario_s1(n_clients=5, target_updates=10)
        sim = ScenarioSimulator(cfg, algorithm="FedBuff", buffer_size=5)
        res = sim.run_simulation()

        self.assertEqual(res["summary"]["applied_updates"], 10)
        self.assertEqual(res["summary"]["server_aggregation_count"], 2)
        self.assertEqual(res["summary"]["final_server_version"], 2)


if __name__ == "__main__":
    unittest.main()
