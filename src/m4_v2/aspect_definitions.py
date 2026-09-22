from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class InterdisciplinaryAspect:
    aspect_id: str
    title: str
    description: str
    anchor_text: str
    target_partner_domains: List[str]


# Standard pedagogical and interdisciplinary curriculum aspects across STEM, Business, and Social Sciences
INTERDISCIPLINARY_ASPECTS: list[InterdisciplinaryAspect] = [
    InterdisciplinaryAspect(
        aspect_id="ethics_governance_law",
        title="Ethics, Governance, Law & Policy",
        description="Explores ethical responsibilities, regulatory compliance, intellectual property, data privacy, and societal policy.",
        anchor_text=(
            "artificial intelligence ethics ethical implications AI regulation "
            "algorithmic governance law intellectual property legal compliance "
            "data privacy accountability policy jurisprudence human rights"
        ),
        target_partner_domains=["Law", "Social Sciences", "Politics", "Philosophy"],
    ),
    InterdisciplinaryAspect(
        aspect_id="economic_policy_finance_trade",
        title="Macroeconomic Policy, Global Trade & Financial Systems",
        description="Explores fiscal and monetary policy, central banking, international trade, economic development, and capital market dynamics.",
        anchor_text=(
            "economics macroeconomics microeconomics economic policy fiscal policy "
            "monetary policy global trade international economics financial systems "
            "banking development economics econometrics public policy international finance"
        ),
        target_partner_domains=["Politics", "Law", "Business School", "Economics"],
    ),
    InterdisciplinaryAspect(
        aspect_id="behavioural_decision_making_economics",
        title="Behavioural Economics & Decision Sciences",
        description="Investigates psychological biases in market decision-making, game theory, consumer behaviour, and experimental economics.",
        anchor_text=(
            "behavioural economics consumer behaviour decision making game theory "
            "experimental economics market psychology cognitive biases bounded rationality "
            "economic experiments social preferences psychology of choice"
        ),
        target_partner_domains=["Psychology", "Life Sciences", "Economics and Finance"],
    ),
    InterdisciplinaryAspect(
        aspect_id="visual_perception_design_hci",
        title="Visual Perception, Graphic Storytelling & HCI",
        description="Addresses user experience design, cognitive visual perception, dashboard aesthetics, and communication through visual narrative.",
        anchor_text=(
            "visual perception graphic design human computer interaction HCI "
            "user experience UX user interface UI data visualisation visual storytelling "
            "infographics dashboard aesthetics typography visual communication"
        ),
        target_partner_domains=["Design", "Arts", "Media and Communications", "Computer Science"],
    ),
    InterdisciplinaryAspect(
        aspect_id="computational_modelling_numerical_sim",
        title="Computational Modeling, Numerical Simulation & Algorithms",
        description="Covers numerical algorithms, scientific programming, mathematical modeling, and computational simulation of physical systems.",
        anchor_text=(
            "computational modeling numerical methods scientific computing algorithms "
            "mathematical simulation Python programming differential equations applied mathematics "
            "computational science mathematical analysis"
        ),
        target_partner_domains=["Mathematics", "Computer Science", "Electronic Engineering"],
    ),
    InterdisciplinaryAspect(
        aspect_id="robotics_hardware_embedded",
        title="Robotics, Embedded Systems & Hardware Integration",
        description="Provides engineering grounding in physical actuators, sensor networks, embedded microcontrollers, and mechatronic systems.",
        anchor_text=(
            "robotics autonomous systems hardware mechatronics embedded systems "
            "sensor networks microcontrollers IoT internet of things physical control "
            "aerospace mechanical automation actuators"
        ),
        target_partner_domains=["Mechanical Engineering", "Electronic Engineering", "Aerospace Engineering"],
    ),
    InterdisciplinaryAspect(
        aspect_id="business_analytics_strategy",
        title="Business Analytics, Enterprise Strategy & Management",
        description="Focuses on technology adoption in enterprise, market dynamics, commercialisation, ROI, and executive decision-making.",
        anchor_text=(
            "business analytics strategy enterprise management technology adoption "
            "commercialisation executive decision making economics finance marketing "
            "digital transformation business model innovation"
        ),
        target_partner_domains=["Business School", "Management", "Economics and Finance"],
    ),
    InterdisciplinaryAspect(
        aspect_id="healthcare_biomedical_applications",
        title="Healthcare & Biomedical Applications",
        description="Applies computational and analytical methods to clinical diagnosis, health informatics, epidemiology, and patient outcomes.",
        anchor_text=(
            "healthcare clinical applications health informatics biomedical engineering "
            "medicine public health epidemiology medical imaging genetics patient diagnostics "
            "pharmacology healthcare analytics"
        ),
        target_partner_domains=["Life Sciences", "Health Sciences", "Medicine", "Biomedical Sciences"],
    ),
    InterdisciplinaryAspect(
        aspect_id="environmental_sustainability_climate",
        title="Environmental Sustainability & Clean Tech",
        description="Integrates climate impact analysis, renewable resources, circular economy, and ecological system modeling.",
        anchor_text=(
            "environmental sustainability clean technology renewable energy climate change "
            "carbon footprint ecological modeling green computing circular economy "
            "environmental protection sustainable engineering"
        ),
        target_partner_domains=["Civil and Environmental Engineering", "Life Sciences", "Mechanical Engineering"],
    ),
    InterdisciplinaryAspect(
        aspect_id="cybersecurity_trust_cryptography",
        title="Cybersecurity, Cryptography & Digital Trust",
        description="Examines security vulnerabilities, cryptographic protocols, defense mechanisms, and secure system architecture.",
        anchor_text=(
            "cybersecurity cryptography digital trust security vulnerabilities "
            "network security threat modeling privacy encryption security audit "
            "cyber defense secure architecture"
        ),
        target_partner_domains=["Computer Science", "Electronic Engineering", "Mathematics"],
    ),
    InterdisciplinaryAspect(
        aspect_id="behavioural_psychology_cognition",
        title="Behavioural Psychology & Cognitive Science",
        description="Investigates human decision biases, cognitive ergonomics, psychology of technology users, and neurocognitive factors.",
        anchor_text=(
            "cognitive psychology behavioural science human decision making cognition "
            "user perception neuroscience cognitive ergonomics psychometrics "
            "social psychology human factors"
        ),
        target_partner_domains=["Life Sciences", "Psychology", "Social Sciences"],
    ),
]
