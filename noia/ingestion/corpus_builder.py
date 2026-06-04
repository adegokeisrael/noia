"""
noia/ingestion/corpus_builder.py
─────────────────────────────────
Synthetic telecom operational corpus generator for NOIA Build-a-thon.

Produces realistic documents across four knowledge categories:
  - Runbooks / Standard Operating Procedures (SOPs)
  - Incident records
  - SLA clause documents
  - Maintenance logs

All documents are written to the configured output directory as plain-text
files with structured metadata headers.

Usage (CLI)::

    python -m noia.ingestion.corpus_builder --output data/corpus --count 500

Usage (API)::

    from noia.ingestion.corpus_builder import CorpusBuilder
    builder = CorpusBuilder(output_dir="data/corpus")
    builder.build(total=500)
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console
from rich.progress import track

console = Console()
app = typer.Typer(help="Generate synthetic telecom operational corpus.")

# ── Document type distribution ──────────────────────────────────────────────
DISTRIBUTION: dict[str, float] = {
    "runbook": 0.24,
    "incident": 0.40,
    "sla": 0.16,
    "maintenance": 0.20,
}

# ── Domain vocabulary ────────────────────────────────────────────────────────
VENDORS = ["Nokia", "Ericsson", "Huawei", "Cisco", "Juniper", "ZTE"]
PROTOCOLS = ["BGP", "OSPF", "IS-IS", "MPLS", "LDP", "RSVP-TE", "BFD", "VRRP", "STP"]
INTERFACES = ["GigabitEthernet", "TenGigE", "HundredGigE", "LoopBack", "Bundle-Ether"]
FAULT_TYPES = [
    "route flap", "fiber cut", "optical power degradation", "hardware failure",
    "CPU overload", "memory exhaustion", "interface CRC errors", "BGP session drop",
    "OSPF adjacency loss", "MPLS label exhaustion", "power supply failure",
    "cooling unit alarm", "clock synchronisation loss", "packet loss spike",
    "latency anomaly", "DNS resolution failure", "NTP desynchronisation",
]
NODES = [f"PE-{i:02d}" for i in range(1, 21)] + \
        [f"P-{i:02d}" for i in range(1, 11)] + \
        [f"CE-{i:02d}" for i in range(1, 16)]
REGIONS = ["Lagos", "Accra", "Nairobi", "Cairo", "Dakar", "Abidjan",
           "Kigali", "Dar es Salaam", "Johannesburg", "Addis Ababa"]
PRIORITIES = ["P1", "P2", "P3"]
ENGINEERS = [
    "Eng. Osei", "Eng. Diallo", "Eng. Okeke", "Eng. Al-Rashid",
    "Eng. Benkhadda", "Eng. Mensah", "Eng. Nwosu", "Eng. Hassan",
]

# ── Runbook templates ────────────────────────────────────────────────────────
RUNBOOK_TEMPLATES = [
    {
        "title": "BGP Session Recovery Procedure — {vendor} {node}",
        "body": (
            "SCOPE: This procedure applies to all BGP session failures on {vendor} "
            "{node} equipment in the {region} region.\n\n"
            "PRECONDITIONS:\n"
            "  1. Obtain change authorisation from NOC Team Lead.\n"
            "  2. Confirm customer SLA obligations: availability ≥ {sla_avail}%.\n"
            "  3. Open incident ticket in ServiceNow and record change window.\n\n"
            "PROCEDURE:\n"
            "  Step 1 — Verify BGP neighbour state:\n"
            "    > show bgp neighbors {peer_ip} | include BGPstate\n"
            "    Expected: BGPstate=Established. If 'Active' or 'Idle', proceed.\n\n"
            "  Step 2 — Check interface status:\n"
            "    > show interfaces {iface} brief\n"
            "    Verify line protocol is UP. If DOWN, escalate to transport team.\n\n"
            "  Step 3 — Inspect BGP timers:\n"
            "    > show bgp neighbors {peer_ip} | include hold|keepalive\n"
            "    If hold timer ≤ 10s, negotiate new timer values with peer operator.\n\n"
            "  Step 4 — Clear BGP session (with authorisation):\n"
            "    > clear bgp {peer_ip} soft\n"
            "    Monitor for re-establishment over 3 minutes.\n\n"
            "  Step 5 — Validate route advertisement:\n"
            "    > show bgp ipv4 unicast neighbors {peer_ip} advertised-routes\n"
            "    Confirm prefixes are being advertised correctly.\n\n"
            "  Step 6 — Update incident ticket with resolution steps and close.\n\n"
            "ESCALATION: If session does not recover within 15 minutes, "
            "escalate to Level-3 Network Engineering.\n\n"
            "REFERENCES: {vendor} BGP Configuration Guide v{version}, RFC 4271."
        ),
    },
    {
        "title": "MPLS LSP Re-optimisation Procedure — {region} Core",
        "body": (
            "SCOPE: Re-optimisation of RSVP-TE Label Switched Paths (LSPs) "
            "following link failure or planned maintenance in the {region} core.\n\n"
            "PRECONDITIONS:\n"
            "  1. Confirm RSVP-TE topology is converged (OSPF/IS-IS adjacency stable).\n"
            "  2. Verify MPLS traffic engineering database is consistent on {node}.\n\n"
            "PROCEDURE:\n"
            "  Step 1 — Identify affected LSPs:\n"
            "    > show mpls traffic-eng tunnels brief\n"
            "    Filter for LSPs with state 'DOWN' or 'REOPTIMISING'.\n\n"
            "  Step 2 — Trigger manual re-optimisation:\n"
            "    > mpls traffic-eng reoptimize tunnel-te {tunnel_id}\n\n"
            "  Step 3 — Verify new path:\n"
            "    > show mpls traffic-eng tunnels tunnel-te {tunnel_id} detail\n"
            "    Confirm explicit route and bandwidth reservation.\n\n"
            "  Step 4 — Monitor for 10 minutes. If LSP remains DOWN, "
            "check RSVP Path messages and PCEP connectivity.\n\n"
            "ESCALATION: MPLS team on-call: mpls-oncall@operator.net"
        ),
    },
    {
        "title": "Optical Power Degradation — OTN Link Recovery on {node}",
        "body": (
            "FAULT TYPE: Optical power degradation / LOS (Loss of Signal) alarm.\n"
            "AFFECTED NODE: {node}, Region: {region}.\n\n"
            "PROCEDURE:\n"
            "  Step 1 — Identify alarming port:\n"
            "    > show alarms detail | include LOS|OPR|OPT\n\n"
            "  Step 2 — Measure receive power (OPR):\n"
            "    > show controller optics {iface}\n"
            "    Acceptable range: {opr_min} dBm to {opr_max} dBm.\n"
            "    If OPR < {opr_min} dBm, suspect fiber degradation.\n\n"
            "  Step 3 — Perform ORL (Optical Return Loss) test with OTDR "
            "from nearest splice point.\n\n"
            "  Step 4 — If OTDR confirms break, dispatch field crew to GPS "
            "coordinates {gps}. Expected restoration time: {eta} hours.\n\n"
            "  Step 5 — Update affected customer SLAs and trigger SLA breach "
            "notification if outage exceeds {breach_threshold} minutes.\n\n"
            "RELATED INCIDENTS: See incident tag #OTN-{region.upper()}"
        ),
    },
]

# ── Incident templates ───────────────────────────────────────────────────────
INCIDENT_TEMPLATES = [
    {
        "title": "{priority} Incident — {fault} on {node} ({region})",
        "body": (
            "INCIDENT ID: INC-{inc_id}\n"
            "Priority: {priority}\n"
            "Status: {status}\n"
            "Opened: {opened}\n"
            "Closed: {closed}\n"
            "Assigned To: {engineer}\n\n"
            "DESCRIPTION:\n"
            "At {opened}, monitoring system triggered a {priority} alert for "
            "{fault} on node {node} in the {region} region. The fault affected "
            "{customers} customers with an estimated revenue impact of "
            "USD {revenue_impact}/hour.\n\n"
            "TIMELINE:\n"
            "  {t0}  — Alert received from NMS. Auto-ticket created.\n"
            "  {t1}  — NOC engineer {engineer} acknowledged and began investigation.\n"
            "  {t2}  — Root cause identified: {root_cause}.\n"
            "  {t3}  — Remediation action initiated: {action}.\n"
            "  {t4}  — Service restored. Customer impact cleared.\n"
            "  {closed}  — Incident closed. Post-incident review scheduled.\n\n"
            "ROOT CAUSE ANALYSIS:\n"
            "{rca_detail}\n\n"
            "RESOLUTION:\n"
            "{resolution}\n\n"
            "LESSONS LEARNED:\n"
            "  1. {lesson_1}\n"
            "  2. {lesson_2}\n\n"
            "SLA IMPACT:\n"
            "  Downtime: {downtime} minutes.\n"
            "  SLA threshold: {sla_threshold} minutes/month.\n"
            "  SLA credit triggered: {sla_credit}.\n\n"
            "RELATED RUNBOOKS: {runbook_ref}\n"
            "RELATED INCIDENTS: {related_incidents}"
        ),
    },
]

# ── SLA templates ────────────────────────────────────────────────────────────
SLA_TEMPLATES = [
    {
        "title": "SLA Schedule — {customer} — {service_type} Service",
        "body": (
            "SERVICE LEVEL AGREEMENT — SCHEDULE {schedule_id}\n"
            "Customer: {customer}\n"
            "Service Type: {service_type}\n"
            "Effective Date: {effective_date}\n"
            "Review Date: {review_date}\n\n"
            "1. AVAILABILITY COMMITMENT\n"
            "   The Operator commits to a monthly network availability of "
            "{availability}% measured at the network demarcation point.\n"
            "   Measurement period: calendar month.\n"
            "   Permitted downtime per month: {permitted_downtime} minutes.\n\n"
            "2. LATENCY\n"
            "   One-way propagation latency (95th percentile): ≤ {latency_ms} ms.\n"
            "   Measurement: continuous probe every 60 seconds.\n\n"
            "3. PACKET LOSS\n"
            "   Monthly average packet loss rate: ≤ {packet_loss}%.\n\n"
            "4. MEAN TIME TO REPAIR (MTTR)\n"
            "   P1 (Critical): ≤ {mttr_p1} hours from fault notification.\n"
            "   P2 (Major):    ≤ {mttr_p2} hours from fault notification.\n"
            "   P3 (Minor):    ≤ {mttr_p3} hours from fault notification.\n\n"
            "5. ESCALATION PATH\n"
            "   L1 NOC Engineer → L2 Senior NOC → L3 Network Engineering "
            "→ VP Network Operations.\n"
            "   Escalation trigger: fault unresolved after {escalation_trigger} minutes.\n\n"
            "6. CREDITS\n"
            "   Availability breach < {credit_tier_1}%: {credit_pct_1}% monthly fee credit.\n"
            "   Availability breach < {credit_tier_2}%: {credit_pct_2}% monthly fee credit.\n\n"
            "7. EXCLUSIONS\n"
            "   Force majeure events, scheduled maintenance (≥ 72h notice), "
            "customer-side faults."
        ),
    },
]

# ── Maintenance log templates ────────────────────────────────────────────────
MAINTENANCE_TEMPLATES = [
    {
        "title": "Maintenance Record — {maint_id} — {description}",
        "body": (
            "MAINTENANCE ID: {maint_id}\n"
            "Type: {maint_type}\n"
            "Scheduled Start: {start}\n"
            "Scheduled End: {end}\n"
            "Actual Start: {actual_start}\n"
            "Actual End: {actual_end}\n"
            "Affected Nodes: {nodes}\n"
            "Region: {region}\n"
            "Lead Engineer: {engineer}\n\n"
            "DESCRIPTION:\n"
            "{description}\n\n"
            "CUSTOMER IMPACT:\n"
            "  Affected services: {affected_services}\n"
            "  Notifications sent: {notification_count} customers notified "
            "{notification_lead}h in advance.\n\n"
            "PROCEDURE EXECUTED:\n"
            "{procedure}\n\n"
            "OUTCOME:\n"
            "{outcome}\n\n"
            "ISSUES ENCOUNTERED:\n"
            "{issues}\n\n"
            "FOLLOW-UP ACTIONS:\n"
            "  1. {followup_1}\n"
            "  2. {followup_2}\n\n"
            "SIGN-OFF: {engineer} | {actual_end}"
        ),
    },
]


class CorpusBuilder:
    """
    Generates a synthetic telecom operational corpus for NOIA indexing.

    Parameters
    ----------
    output_dir : str | Path
        Directory to write generated documents.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, output_dir: str | Path = "data/corpus", seed: int = 42) -> None:
        self.output_dir = Path(output_dir)
        self.rng = random.Random(seed)

    def _rand_ip(self) -> str:
        return ".".join(str(self.rng.randint(1, 254)) for _ in range(4))

    def _rand_date(self, start: datetime, days: int = 365) -> datetime:
        return start + timedelta(days=self.rng.randint(0, days),
                                 hours=self.rng.randint(0, 23),
                                 minutes=self.rng.randint(0, 59))

    def _generate_runbook(self, doc_id: int) -> dict:
        tpl = self.rng.choice(RUNBOOK_TEMPLATES)
        vendor = self.rng.choice(VENDORS)
        node = self.rng.choice(NODES)
        region = self.rng.choice(REGIONS)
        iface = f"{self.rng.choice(INTERFACES)}0/0/{self.rng.randint(0,3)}"
        peer_ip = self._rand_ip()
        return {
            "doc_id": f"RB-{doc_id:04d}",
            "source_type": "runbook",
            "title": tpl["title"].format(vendor=vendor, node=node, region=region),
            "body": tpl["body"].format(
                vendor=vendor, node=node, region=region,
                iface=iface, peer_ip=peer_ip,
                sla_avail=self.rng.choice(["99.9", "99.95", "99.99"]),
                version=f"{self.rng.randint(5,12)}.{self.rng.randint(0,4)}",
                tunnel_id=self.rng.randint(100, 999),
                opr_min=round(-28 + self.rng.uniform(-2, 2), 1),
                opr_max=round(-8 + self.rng.uniform(-1, 1), 1),
                gps=f"{self.rng.uniform(-10, 15):.4f}N, {self.rng.uniform(-5, 45):.4f}E",
                eta=self.rng.randint(2, 8),
                breach_threshold=self.rng.choice([30, 60, 120]),
            ),
            "timestamp": self._rand_date(datetime(2023, 1, 1)).isoformat(),
            "version": f"v{self.rng.randint(1,5)}.{self.rng.randint(0,9)}",
        }

    def _generate_incident(self, doc_id: int) -> dict:
        tpl = INCIDENT_TEMPLATES[0]
        fault = self.rng.choice(FAULT_TYPES)
        node = self.rng.choice(NODES)
        region = self.rng.choice(REGIONS)
        priority = self.rng.choice(PRIORITIES)
        opened_dt = self._rand_date(datetime(2023, 6, 1), 365)
        duration = timedelta(minutes=self.rng.randint(15, 480))
        t = [opened_dt + timedelta(minutes=i * self.rng.randint(5, 30)) for i in range(5)]
        closed_dt = opened_dt + duration
        rca_options = [
            f"The {fault} was caused by a misconfigured {self.rng.choice(PROTOCOLS)} timer following a recent software upgrade.",
            f"Physical layer degradation on {node} triggered by high ambient temperature in the {region} data centre.",
            f"A configuration drift introduced during a maintenance window caused {fault} due to incorrect ACL entries.",
            f"Capacity exhaustion on the {node}-{self.rng.choice(NODES)} link led to {fault} under peak traffic load.",
        ]
        return {
            "doc_id": f"INC-{doc_id:04d}",
            "source_type": "incident",
            "title": tpl["title"].format(priority=priority, fault=fault, node=node, region=region),
            "body": tpl["body"].format(
                inc_id=f"{doc_id:06d}",
                priority=priority, status="Closed",
                opened=opened_dt.strftime("%Y-%m-%d %H:%M UTC"),
                closed=closed_dt.strftime("%Y-%m-%d %H:%M UTC"),
                engineer=self.rng.choice(ENGINEERS),
                fault=fault, node=node, region=region,
                customers=self.rng.randint(50, 5000),
                revenue_impact=self.rng.randint(500, 50000),
                t0=t[0].strftime("%H:%M"), t1=t[1].strftime("%H:%M"),
                t2=t[2].strftime("%H:%M"), t3=t[3].strftime("%H:%M"),
                t4=t[4].strftime("%H:%M"),
                root_cause=self.rng.choice(["hardware fault", "software bug", "configuration error", "external factor"]),
                action=f"Applied {self.rng.choice(RUNBOOK_TEMPLATES)['title'].split('—')[0].strip()} procedure",
                rca_detail=self.rng.choice(rca_options),
                resolution=f"Restored service by applying {self.rng.choice(['configuration rollback', 'software patch', 'hardware replacement', 'traffic rerouting'])}.",
                lesson_1="Automated pre-change configuration validation should be enforced.",
                lesson_2="Temperature monitoring thresholds should be tightened for summer months.",
                downtime=int(duration.total_seconds() // 60),
                sla_threshold=self.rng.choice([30, 60, 120, 240]),
                sla_credit=self.rng.choice(["Yes — 5% credit applied", "No — within threshold"]),
                runbook_ref=f"RB-{self.rng.randint(1, 120):04d}",
                related_incidents=", ".join(f"INC-{self.rng.randint(1, 500):06d}" for _ in range(2)),
            ),
            "timestamp": opened_dt.isoformat(),
            "priority": priority,
        }

    def _generate_sla(self, doc_id: int) -> dict:
        tpl = SLA_TEMPLATES[0]
        customers = ["TelecomCo Ghana", "NairobiFibre Ltd", "Lagos Broadband Services",
                     "CairoNet ISP", "SenegaSat", "KigaliConnect"]
        service_types = ["MPLS L3VPN", "Metro Ethernet", "IP Transit",
                         "Dedicated Internet Access", "SD-WAN Managed Service"]
        avail = self.rng.choice(["99.9", "99.95", "99.99"])
        eff_date = self._rand_date(datetime(2022, 1, 1), 730)
        return {
            "doc_id": f"SLA-{doc_id:04d}",
            "source_type": "sla",
            "title": tpl["title"].format(
                customer=self.rng.choice(customers),
                service_type=self.rng.choice(service_types)
            ),
            "body": tpl["body"].format(
                schedule_id=f"SCH-{doc_id:04d}",
                customer=self.rng.choice(customers),
                service_type=self.rng.choice(service_types),
                availability=avail,
                permitted_downtime=round((1 - float(avail) / 100) * 43200, 1),
                latency_ms=self.rng.choice([5, 10, 20, 50]),
                packet_loss=self.rng.choice(["0.01", "0.05", "0.1"]),
                mttr_p1=4, mttr_p2=8, mttr_p3=24,
                escalation_trigger=self.rng.choice([30, 45, 60]),
                credit_tier_1=round(float(avail) - 0.5, 2),
                credit_tier_2=round(float(avail) - 1.0, 2),
                credit_pct_1=10, credit_pct_2=25,
                effective_date=eff_date.strftime("%Y-%m-%d"),
                review_date=(eff_date + timedelta(days=365)).strftime("%Y-%m-%d"),
            ),
            "timestamp": eff_date.isoformat(),
        }

    def _generate_maintenance(self, doc_id: int) -> dict:
        tpl = MAINTENANCE_TEMPLATES[0]
        maint_types = ["Planned Software Upgrade", "Hardware Replacement",
                       "Fibre Splice Repair", "Capacity Augmentation",
                       "Security Patch Deployment", "Battery Backup Test"]
        node_count = self.rng.randint(1, 4)
        affected_nodes = ", ".join(self.rng.sample(NODES, node_count))
        region = self.rng.choice(REGIONS)
        start_dt = self._rand_date(datetime(2023, 1, 1), 365)
        planned_dur = timedelta(hours=self.rng.randint(2, 8))
        actual_dur = planned_dur + timedelta(minutes=self.rng.randint(-30, 120))
        desc = self.rng.choice(maint_types)
        return {
            "doc_id": f"MNT-{doc_id:04d}",
            "source_type": "maintenance",
            "title": tpl["title"].format(maint_id=f"CHG-{doc_id:05d}", description=desc),
            "body": tpl["body"].format(
                maint_id=f"CHG-{doc_id:05d}",
                maint_type=desc,
                start=start_dt.strftime("%Y-%m-%d %H:%M UTC"),
                end=(start_dt + planned_dur).strftime("%Y-%m-%d %H:%M UTC"),
                actual_start=start_dt.strftime("%Y-%m-%d %H:%M UTC"),
                actual_end=(start_dt + actual_dur).strftime("%Y-%m-%d %H:%M UTC"),
                nodes=affected_nodes, region=region,
                engineer=self.rng.choice(ENGINEERS),
                description=f"{desc} on nodes {affected_nodes} in the {region} region.",
                affected_services=self.rng.choice(["MPLS L3VPN", "Metro Ethernet", "IP Transit"]),
                notification_count=self.rng.randint(5, 200),
                notification_lead=self.rng.choice([24, 48, 72]),
                procedure=(
                    "  1. Take pre-change configuration backup.\n"
                    "  2. Execute maintenance procedure per approved change record.\n"
                    "  3. Validate service restoration.\n"
                    "  4. Confirm with affected customers.\n"
                    "  5. Close change record."
                ),
                outcome=self.rng.choice([
                    "Completed successfully within maintenance window.",
                    "Completed with 45-minute overrun due to unexpected configuration complexity.",
                    "Partial completion — secondary task deferred to next window.",
                ]),
                issues=self.rng.choice([
                    "None.",
                    "Minor configuration discrepancy identified and corrected.",
                    "One service required manual restart post-upgrade.",
                ]),
                followup_1="Update runbook with observed configuration differences.",
                followup_2="Schedule post-maintenance monitoring for 72 hours.",
            ),
            "timestamp": start_dt.isoformat(),
        }

    def build(self, total: int = 500) -> list[Path]:
        """
        Generate ``total`` documents and write them to ``output_dir``.

        Returns a list of written file paths.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Compute counts per type
        counts = {k: int(total * v) for k, v in DISTRIBUTION.items()}
        counts["incident"] += total - sum(counts.values())  # absorb rounding remainder

        generators: dict[str, callable] = {
            "runbook":     self._generate_runbook,
            "incident":    self._generate_incident,
            "sla":         self._generate_sla,
            "maintenance": self._generate_maintenance,
        }

        written: list[Path] = []
        global_id = 1

        for doc_type, count in counts.items():
            type_dir = self.output_dir / doc_type
            type_dir.mkdir(exist_ok=True)
            for i in track(range(1, count + 1),
                           description=f"[cyan]Generating {doc_type}s...[/cyan]"):
                doc = generators[doc_type](global_id)
                doc["source_type"] = doc_type
                out_path = type_dir / f"{doc['doc_id']}.txt"
                with out_path.open("w", encoding="utf-8") as fh:
                    fh.write(f"DOC_ID: {doc['doc_id']}\n")
                    fh.write(f"SOURCE_TYPE: {doc_type}\n")
                    fh.write(f"TITLE: {doc['title']}\n")
                    fh.write(f"TIMESTAMP: {doc['timestamp']}\n")
                    fh.write("─" * 72 + "\n\n")
                    fh.write(doc["body"])
                    fh.write("\n")
                # Save sidecar metadata
                meta_path = type_dir / f"{doc['doc_id']}.meta.json"
                meta = {k: str(v) if isinstance(v, Path) else v
                        for k, v in doc.items() if k != "body"}
                meta["file_path"] = str(out_path)
                with meta_path.open("w", encoding="utf-8") as fh:
                    json.dump(meta, fh, indent=2)
                written.append(out_path)
                global_id += 1

        console.print(f"[bold green]✓ Corpus built:[/bold green] {len(written)} documents in {self.output_dir}")
        return written


@app.command()
def main(
    output: Path = typer.Option(Path("data/corpus"), help="Output directory."),
    count: int = typer.Option(500, help="Total number of documents to generate."),
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    """Generate synthetic telecom operational corpus for NOIA."""
    builder = CorpusBuilder(output_dir=output, seed=seed)
    builder.build(total=count)


if __name__ == "__main__":
    app()
