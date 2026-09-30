import random
import re

from research.noise import FAMILIES, perturb


def test_noise_keeps_numbers_and_letter_values():
    text = "В соответствии с пп. «а» п. 2 ч. 1 ст. 19.5 КоАП РФ, статьей 81 Трудового кодекса Российской Федерации."
    for family in [*FAMILIES, "all"]:
        for seed in range(20):
            noisy = perturb(text, family, random.Random(seed))
            assert re.findall(r"\d+", noisy) == re.findall(r"\d+", text), (family, noisy)
            assert "«а»" in noisy or "«А»" in noisy, (family, noisy)
