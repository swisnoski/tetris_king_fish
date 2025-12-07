from setuptools import find_packages, setup

package_name = "tetris_king"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="swisnoski",
    maintainer_email="swisnoski@olin.edu",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "tetris_arm = tetris_king.tetris_arm:main",
            "mock_decision = tetris_king.mock_decision:main",
        ],
    },
)
