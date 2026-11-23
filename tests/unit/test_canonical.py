"""Scorer v5 (canonical comparison): presentation-only differences are equal, real errors are not."""

from __future__ import annotations

import pytest

from resume_bench.grading.canonical import score_resume_v5
from resume_bench.grading.grader import grade_single_strict
from resume_bench.grading.strict import score_resume_strict
from resume_bench.schema.sections import SECTIONS


def _edu(study=None, area=None):
    return {"education": [{"institution": "MIT", "studyType": study, "area": area}]}


def _units(result, section):
    s = result["sections"][section]
    return s["correct"], s["expected"], s["predicted"]


@pytest.mark.parametrize(
    ("label", "pred", "units"),
    [
        (_edu("Bachelor's", "Computer Science"), _edu("Bachelors", "Computer Science"), (3, 3, 3)),
        (_edu("Bachelor of Science in Computer Science"), _edu("B.S.", "Computer Science"), (2, 2, 2)),
        (_edu("Bachelors in Computer Science"), _edu("B.S. in Computer Science"), (2, 2, 2)),
        (_edu("Bachelor of Science in Computer Science"), _edu("Bachelor of Science in Biology"), (1, 2, 2)),
        (_edu("Bachelor of Science", "Computer Science"), _edu("Bachelor of Arts", "Computer Science"), (2, 3, 3)),
        (_edu("Semester 1"), _edu("Semester 2"), (1, 2, 2)),
        (_edu("Executive MBA"), _edu("MBA"), (1, 2, 2)),
    ],
)
def test_degree(label, pred, units):
    assert _units(score_resume_v5(label, pred), "education") == units


def test_all_caps_styling_is_equal_but_other_case_is_not():
    lab = {"experience": [{"company": "ACME CORP", "position": "Engineer"}]}
    same = {"experience": [{"company": "Acme Corp", "position": "Engineer"}]}
    assert score_resume_v5(lab, same)["f1"] == 1.0
    lab2 = {"experience": [{"company": "Acme corp", "position": "Engineer"}]}
    assert score_resume_v5(lab2, same)["f1"] < 1.0


def test_bullet_markers_and_emoji_are_ignored():
    lab = {"experience": [{"company": "A", "position": "B", "description": ["Led team", "Built X"]}]}
    out = {"experience": [{"company": "A", "position": "B", "description": ["• Led team", "🚀 Built X"]}]}
    assert score_resume_v5(lab, out)["f1"] == 1.0


def test_name_order_is_presentation():
    lab = {"basics": {"fname": "Wei", "lname": "DENG"}}
    out = {"basics": {"fname": "DENG", "lname": "Wei"}}
    assert score_resume_v5(lab, out)["f1"] == 1.0
    assert score_resume_strict(lab, out, SECTIONS)["f1"] < 1.0


def test_location_slots_are_a_set():
    lab = {"basics": {"fname": "A", "city": "Singapore"}}
    out = {"basics": {"fname": "A", "country": "Singapore"}}
    assert score_resume_v5(lab, out)["f1"] == 1.0


def test_wrong_text_still_wrong():
    lab = {"experience": [{"company": "Acme", "position": "Engineer", "description": ["Built APIs"]}]}
    out = {"experience": [{"company": "Acme", "position": "Engineer", "description": ["Built a website"]}]}
    assert score_resume_v5(lab, out)["f1"] < 1.0


def test_identical_scores_one_and_released_scorer_untouched():
    doc = {"basics": {"fname": "Alice", "lname": "Smith"}, "skills": [{"skills": ["Python", "SQL"]}]}
    assert score_resume_v5(doc, doc)["f1"] == 1.0
    before = score_resume_strict(
        {"basics": {"fname": "Wei", "lname": "DENG"}}, {"basics": {"fname": "DENG", "lname": "Wei"}}, SECTIONS
    )
    score_resume_v5(doc, doc)
    after = score_resume_strict(
        {"basics": {"fname": "Wei", "lname": "DENG"}}, {"basics": {"fname": "DENG", "lname": "Wei"}}, SECTIONS
    )
    assert before == after


def test_grader_scorer_switch():
    lab = {"basics": {"fname": "Wei", "lname": "DENG"}}
    out = {"basics": {"fname": "DENG", "lname": "Wei"}}
    assert grade_single_strict(lab, out)["f1"] == 1.0
    assert grade_single_strict(lab, out, scorer="strict")["f1"] < 1.0
    with pytest.raises(ValueError):
        grade_single_strict(lab, out, scorer="nope")


def test_personal_summary_string_prediction():
    lab = {"personalSummary": {"text": "Engineer with 10 years."}}
    assert score_resume_v5(lab, {"personalSummary": "Engineer with 10 years."})["f1"] == 1.0


def test_real_schema_converts_to_openai_strict():
    from resume_bench.providers.openai import _make_strict_schema
    from resume_bench.schema import get_schema

    bad = []

    def walk(node, path="$"):
        if isinstance(node, dict):
            if "properties" in node:
                if node.get("additionalProperties") is not False or set(node.get("required", [])) != set(
                    node["properties"]
                ):
                    bad.append(path)
                for k, v in node["properties"].items():
                    walk(v, f"{path}.{k}")
            if "items" in node:
                walk(node["items"], path + "[]")

    walk(_make_strict_schema(get_schema()))
    assert bad == []


def test_non_string_values_do_not_crash():
    lab = {"education": [{"institution": "MIT", "studyType": "BS"}]}
    out = {"education": [{"institution": "MIT", "studyType": 12, "area": 3.5}]}
    assert 0.0 <= score_resume_v5(lab, out)["f1"] <= 1.0


def test_missing_or_invented_degree_is_not_free():
    lab = _edu("Master of Science", "Computer Science")
    assert _units(score_resume_v5(lab, _edu(None, "Computer Science")), "education") == (2, 3, 2)
    assert _units(score_resume_v5(_edu(None, "Physics"), _edu("PhD", "Physics")), "education") == (2, 2, 3)


@pytest.mark.parametrize(
    ("abbr", "full"),
    [
        ("BSN", "Bachelor of Science in Nursing"),
        ("MSN", "Master of Science in Nursing"),
        ("BSBA", "Bachelor of Science in Business Administration"),
    ],
)
def test_abbreviation_with_field_equals_spelled_out(abbr, full):
    assert _units(score_resume_v5(_edu(abbr), _edu(full)), "education") == (2, 2, 2)


def test_major_and_minor_are_not_interchangeable():
    lab = _edu("BA", "Economics, Minor: Statistics")
    assert _units(score_resume_v5(lab, _edu("BA", "Statistics, Minor: Economics")), "education") == (2, 3, 3)
    assert _units(score_resume_v5(lab, _edu("B.A.", "Economics; Minor: Statistics")), "education") == (3, 3, 3)


def test_language_brackets_kept_unless_proficiency():
    from resume_bench.grading.canonical import split_language

    assert split_language("Chinese (Mandarin)", "Native") == ("Chinese (Mandarin)", "Native")
    assert split_language("Hindi (Fluent)", None) == ("Hindi", "Fluent")
    assert split_language("Serbo-Croatian", None) == ("Serbo-Croatian", None)
    lab = {"languages": [{"name": "Chinese (Mandarin)", "level": "Native"}]}
    out = {"languages": [{"name": "Chinese (Cantonese)", "level": "Native"}]}
    assert score_resume_v5(lab, out)["f1"] < 1.0


def _ed(*items):
    return {"education": [{"institution": "Stanford", **it} for it in items]}


def test_extra_entry_at_same_institution_does_not_steal_the_pairing():
    ms = {"studyType": "M.S.", "area": "Computer Science", "score": "3.9"}
    r = score_resume_v5(_ed(ms), _ed({"area": "Graduate Certificate Coursework"}, ms))
    assert _units(r, "education")[0] == 4


def test_double_degree_in_other_order():
    a = {"studyType": "B.S.", "area": "Economics", "startYear": 2016, "endYear": 2020}
    b = {"studyType": "B.A.", "area": "Mathematics", "startYear": 2016, "endYear": 2020}
    r = score_resume_v5(_ed(a, b), _ed(b, a))
    assert r["sections"]["education"]["f1"] == 1.0


def _exp(desc=None, **k):
    return {"experience": [{"company": "A", "position": "B", **({"description": desc} if desc else {}), **k}]}


def test_moved_line_break_is_not_credited():
    lab = _exp(["Led migration of the billing system", "Reduced costs by 20%"])
    out = _exp(["Led migration of the", "billing system Reduced costs by 20%"])
    assert _units(score_resume_v5(lab, out), "experience") == (2, 4, 4)
    merged = _exp(["Led migration of the billing system Reduced costs by 20%"])
    assert _units(score_resume_v5(lab, merged), "experience") == (3, 4, 3)


def test_bullets_written_as_one_paragraph_count_as_a_merge():
    five = ["A one.", "B two.", "C three.", "D four.", "E five."]
    r = score_resume_v5(_exp(five), _exp(roleDescription=" ".join(five)))
    assert _units(r, "experience") == (3, 7, 3)


def test_linkedin_handle_case_and_turkish_caps():
    from resume_bench.grading.canonical import canon_url

    assert canon_url("in/Jane-Doe") == canon_url("in/jane-doe")
    assert canon_url("lnkd.in/AbC") != canon_url("lnkd.in/abc")  # short-link codes keep their case
    assert canon_url("/in/Jane") != canon_url("/in/jane")
    lab, out = {"basics": {"fname": "İLKER YILMAZ"}}, {"basics": {"fname": "İlker Yılmaz"}}
    assert score_resume_v5(lab, out)["f1"] == 1.0


@pytest.mark.parametrize(
    ("abbr", "full"),
    [("BSCS", "Bachelor of Science in Computer Science"), ("MSEE", "Master of Science in Electrical Engineering")],
)
def test_more_field_abbreviations(abbr, full):
    assert _units(score_resume_v5(_edu(abbr), _edu(full)), "education") == (2, 2, 2)
    plain = "Bachelor of Science" if abbr.startswith("B") else "Master of Science"
    assert _units(score_resume_v5(_edu(abbr), _edu(plain)), "education") == (1, 2, 2)


def test_minors_plural_and_order():
    swapped = score_resume_v5(_edu("BA", "Economics, Minors: Statistics"), _edu("BA", "Statistics, Minors: Economics"))
    assert _units(swapped, "education") == (2, 3, 3)
    two = score_resume_v5(
        _edu("BA", "Economics; Minor: Statistics; Minor: Math"), _edu("BA", "Economics; Minor: Math; Minor: Statistics")
    )
    assert _units(two, "education") == (3, 3, 3)


def test_language_level_after_bracketed_name():
    from resume_bench.grading.canonical import split_language

    assert split_language("Chinese (Mandarin) - Native", None) == ("Chinese (Mandarin)", "Native")
    assert split_language("Chinese (Mandarin) (Native)", None) == ("Chinese (Mandarin)", "Native")
    assert split_language("Español (nativo)", None) == ("Español", "nativo")


def test_merge_and_split_with_punctuated_lines():
    lab = _exp(["Did A.", "Did B.", "Did C."])
    assert _units(score_resume_v5(lab, _exp(["Did A. Did B.", "Did C."])), "experience") == (4, 5, 4)
    assert _units(score_resume_v5(_exp(["Did A. Did B.", "Did C."]), lab), "experience") == (4, 4, 5)
    assert _units(score_resume_v5(lab, _exp(["Did C."], roleDescription="Did A. Did B.")), "experience") == (4, 5, 4)


def test_language_level_with_dash_inside_brackets():
    from resume_bench.grading.canonical import split_language

    assert split_language("Spanish (Intermediate - B1)", None) == ("Spanish", "Intermediate - B1")
    assert split_language("English: Native - IELTS 8", None) == ("English", "Native - IELTS 8")


def test_field_abbreviation_has_its_level_for_pairing():
    lab = {
        "education": [
            {"institution": "X", "studyType": "BSCS", "area": "Computer Science", "startYear": 2016, "endYear": 2020}
        ]
    }
    out = {
        "education": [
            {"institution": "X", "studyType": "M.S.", "area": "Computer Science", "startYear": 2016, "endYear": 2020}
        ]
    }
    assert _units(score_resume_v5(lab, out), "education")[0] == 0


def test_turkish_name_order_and_location():
    assert score_resume_v5({"basics": {"fname": "SILA SOYLU"}}, {"basics": {"fname": "Sıla Soylu"}})["f1"] == 1.0
    lab = {"basics": {"fname": "A", "city": "ISTANBUL"}}
    assert score_resume_v5(lab, {"basics": {"fname": "A", "city": "İstanbul"}})["f1"] == 1.0


def test_second_printed_degree_counts():
    two = score_resume_v5(_edu("B.S., MBA"), _edu("B.S.", "Business Administration"))
    assert _units(two, "education")[0] < 2
    assert _units(score_resume_v5(_edu("B.S., MBA"), _edu("B.S., MBA")), "education") == (2, 2, 2)


def test_turkish_language_level_and_dotless_letter():
    lab = {"languages": [{"name": "İngilizce", "level": "İleri"}]}
    out = {"languages": [{"name": "İNGİLİZCE", "level": "İleri"}]}
    assert score_resume_v5(lab, out)["f1"] == 1.0
    a = {"basics": {"fname": "A", "city": "Diyarbakır"}}
    assert score_resume_v5(a, {"basics": {"fname": "A", "city": "Diyarbakir"}})["f1"] < 1.0
    assert score_resume_v5(a, {"basics": {"fname": "A", "city": "DİYARBAKIR"}})["f1"] == 1.0


@pytest.mark.parametrize(
    ("label", "pred"),
    [
        (
            _edu("B.S.", "Business Administration, Certificate in Finance"),
            _edu("B.S. in Business Administration, Certificate in Finance"),
        ),
        (_edu("B.S., MBA"), _edu("B.S.", "MBA")),
        (
            _edu("Bachelor of Science, Master of Science", "Biology"),
            _edu("Bachelor of Science", "Master of Science in Biology"),
        ),
        (_edu("Master's in Business Administration (MBA)"), _edu("MBA")),
        (_edu("Master's in Business Administration (MBA)"), _edu("Master's", "Business Administration (MBA)")),
        (_edu("Juris Doctor (J.D.)"), _edu("J.D.")),
    ],
)
def test_second_degree_and_restated_abbreviation_any_split(label, pred):
    r = score_resume_v5(label, pred)["sections"]["education"]
    assert r["correct"] == r["expected"]


def test_invented_second_degree_in_area_costs_precision():
    assert _units(score_resume_v5(_edu("B.S."), _edu("B.S.", "MBA")), "education") == (2, 2, 3)
