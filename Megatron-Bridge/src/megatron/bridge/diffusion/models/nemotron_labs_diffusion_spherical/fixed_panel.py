# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fixed diagnostic questions, with verbatim Quick400 prompts and stable seeds.

ARC uses the variable-choice adapter, MMLU uses the native boxed MCQ prompt,
and BBH/XSum use the local Quick400 extensions. This small selected panel is
for comparing training snapshots, not estimating benchmark accuracy.
"""

FIXED_PANEL = (
    {
        "sample_id": "arc_easy:00879",
        "task": "arc_easy",
        "prompt": "Answer the following multiple choice question. The last line of your response should be in the "
        'following format: "Answer: X", where X is one of A/B/C/D.\n'
        "\n"
        "A flat piece of paper has a mass of 5 grams. When the paper is crumpled into a ball, the mass "
        "of the crumpled paper is\n"
        "\n"
        "A) much more than that of the flat paper.\n"
        "B) slightly more than that of the flat paper.\n"
        "C) the same as that of the flat paper.\n"
        "D) slightly less than that of the flat paper.",
        "reference_answer": "C",
        "seed": 4274166926,
    },
    {
        "sample_id": "arc_easy:01785",
        "task": "arc_easy",
        "prompt": "Answer the following multiple choice question. The last line of your response should be in the "
        'following format: "Answer: X", where X is one of A/B/C/D.\n'
        "\n"
        "A glass of ice water is placed on a table. After 10 minutes, there are drops of water on the "
        "outside surface of the glass. Which change in phase caused the drops of water?\n"
        "\n"
        "A) Liquid water in the air evaporated into a gas.\n"
        "B) Liquid water in the air condensed into a solid.\n"
        "C) Water vapor from the air evaporated into a liquid.\n"
        "D) Water vapor from the air condensed into a liquid.",
        "reference_answer": "D",
        "seed": 3975656260,
    },
    {
        "sample_id": "arc_challenge:00626",
        "task": "arc_challenge",
        "prompt": "Answer the following multiple choice question. The last line of your response should be in the "
        'following format: "Answer: X", where X is one of A/B/C/D.\n'
        "\n"
        "All stars in the universe go through a life cycle. What factor determines how stars progress "
        "through this cycle?\n"
        "\n"
        "A) the radiation that is present in the star\n"
        "B) the amount of energy the star consumes\n"
        "C) the quantity of mass with which the star began\n"
        "D) the location of the star in the universe",
        "reference_answer": "C",
        "seed": 660153829,
    },
    {
        "sample_id": "mmlu:03245",
        "task": "mmlu",
        "prompt": "Answer the following multiple choice question. The last line of your response should be in the "
        "following format: 'Answer: \\boxed{A/B/C/D}' (e.g. 'Answer: \\boxed{A}').\n"
        "\n"
        "What is the role of the compiler in the process of developing executable software?\n"
        "\n"
        "A) Managing specification files created as part of the development process\n"
        "B) Running and testing the executable created by the programmer\n"
        "C) Tracking older versions of the software in case an error is found and the software needs to "
        "be reverted to an earlier form\n"
        "D) Translating a program written in an abstract, high-level language into a program with the "
        "same behavior expressed in machine code",
        "reference_answer": "D",
        "seed": 1614384161,
    },
    {
        "sample_id": "mmlu:04678",
        "task": "mmlu",
        "prompt": "Answer the following multiple choice question. The last line of your response should be in the "
        "following format: 'Answer: \\boxed{A/B/C/D}' (e.g. 'Answer: \\boxed{A}').\n"
        "\n"
        "Because people with relatively low incomes spend a larger percentage of their income on food "
        "than people with relatively high incomes, a sales tax on food would fall into which category of "
        "taxes?\n"
        "\n"
        "A) Progressive\n"
        "B) Proportional\n"
        "C) Regressive\n"
        "D) Neutral",
        "reference_answer": "C",
        "seed": 1437832084,
    },
    {
        "sample_id": "bbh:logical_deduction_three_objects:00161",
        "task": "bbh",
        "prompt": "Solve the task below. You may reason briefly if needed.\n"
        'End with "Answer: <answer>" on the final line.\n'
        "For multiple-choice tasks use the option letter; otherwise provide only the requested value or "
        "sequence.\n"
        "\n"
        "The following paragraphs each describe a set of three objects arranged in a fixed order. The "
        "statements are logically consistent within each paragraph. A fruit stand sells three fruits: "
        "kiwis, oranges, and watermelons. The oranges are more expensive than the kiwis. The oranges are "
        "less expensive than the watermelons.\n"
        "Options:\n"
        "(A) The kiwis are the most expensive\n"
        "(B) The oranges are the most expensive\n"
        "(C) The watermelons are the most expensive",
        "reference_answer": "(C)",
        "seed": 245139185,
    },
    {
        "sample_id": "bbh:boolean_expressions:00029",
        "task": "bbh",
        "prompt": "Solve the task below. You may reason briefly if needed.\n"
        'End with "Answer: <answer>" on the final line.\n'
        "For multiple-choice tasks use the option letter; otherwise provide only the requested value or "
        "sequence.\n"
        "\n"
        "not False or True and False or False is",
        "reference_answer": "True",
        "seed": 1232616174,
    },
    {
        "sample_id": "xsum:06932",
        "task": "xsum",
        "prompt": "Summarize the following news article in one sentence. Return only the summary.\n"
        "\n"
        "The 45-year-old, who had a spell as a player at Headingley between 1990 and 1995, spent eight "
        "years as Essex coach before leaving at the end of 2015.\n"
        "He replaces Richard Pyrah as Diamonds boss, with the latter to assist new Yorkshire head coach "
        "Andrew Gale.\n"
        '"The opportunity to work with an elite female team is really exciting for me," Grayson told the '
        "club website.\n"
        "Yorkshire Diamonds will start their 2017 Super League campaign with a Roses game against "
        "Lancashire Thunder on 11 August.\n"
        "The two sides won only one game each in the round-robin group stage of the inaugural "
        "competition last summer.",
        "reference_answer": "Yorkshire Diamonds have appointed Paul Grayson as their head coach on a deal until "
        "the end of 2019.",
        "seed": 1174143901,
    },
)
