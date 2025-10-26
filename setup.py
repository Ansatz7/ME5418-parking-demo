from pathlib import Path

from setuptools import find_packages, setup


def read_requirements() -> list[str]:
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        return []
    return [line.strip() for line in requirements_file.read_text(encoding="utf-8").splitlines() if line.strip()]


setup(
    name="parking-project-submission",
    version="0.1.0",
    description="Parking environment demo submission package",
    packages=find_packages(include=["parking_project_submission", "parking_project_submission.*"]),
    include_package_data=True,
    python_requires=">=3.8,<3.13",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": ["parking-gym-demo=parking_project_submission.gym_demo:main"],
    },
)
