"""Degree comparison: equal degrees earn all their units however they are split; different degrees do not."""

from __future__ import annotations

import pytest

from resume_bench.grading.canonical import score_resume_v5

FULL = "full"


def d(st, ar=None):
    return {"education": [{"institution": "X", "studyType": st, "area": ar}]}


CASES = [
    (
        ("B.A. in Political Science, B.S. in Biology",),
        ("Bachelor of Arts in Political Science, Bachelor of Science in Biology",),
        FULL,
    ),
    (("B.A. in Political Science, B.S. in Biology",), ("B.A. in Political Science and Biology",), "notfull"),
    (("B.A.", "Political Science, B.S. in Biology"), ("B.A.", "Political Science, Biology"), "notfull"),
    (("EMBA",), ("MBA",), "notfull"),
    (("Bachelor of Science in Computer Science",), ("Bachelor of Science-Computer Science",), FULL),
    (("Bachelor of Science", "Nursing"), ("Bachelor of Science Nursing",), FULL),
    (("Bachelor of Commerce", "Accounting"), ("Bachelor of Commerce-Accounting",), FULL),
    (("B. S. in Computer Science",), ("B.S. in Computer Science",), FULL),
    (
        ("Bachelor of Science with Honours", "Computer Science"),
        ("Bachelor of Science", "Honours in Computer Science"),
        FULL,
    ),
    (
        ("B.S.", "Business Administration, Certificate in Finance"),
        ("B.S. in Business Administration, Certificate in Finance", None),
        FULL,
    ),
    (
        ("Bachelor of Science", "Accounting, Minor in Finance, Certificate in Data Analytics"),
        ("Bachelor of Science in Accounting, Minor in Finance, Certificate in Data Analytics", None),
        FULL,
    ),
    (("B.S., MBA", None), ("B.S.", "MBA"), FULL),
    (("B.S.", None), ("B.S.", "MBA"), (2, 2, 3)),
    (
        ("Bachelor of Science, Master of Science", "Biology"),
        ("Bachelor of Science", "Master of Science in Biology"),
        FULL,
    ),
    (("Master's in Business Administration (MBA)", None), ("MBA", None), FULL),
    (("Master's in Business Administration (MBA)", None), ("Master's", "Business Administration (MBA)"), FULL),
    (("Juris Doctor (J.D.)", None), ("J.D.", None), FULL),
    (("Master in Computer Applications (MCA)", None), ("MCA", None), FULL),
    (("B.S., MBA", None), ("B.S.", "Business Administration"), "notfull"),
    (("Bachelor of Science (B.S.)", None), ("B.S.", None), FULL),
    (("B.S.", "Computer Science"), ("Bachelor of Science in Computer Science", None), FULL),
    (("Bachelor of Science in Computer Science", "Computer Science"), ("B.S.", "Computer Science"), FULL),
    (("Master", "Computer Science"), ("Master of Computer Science", None), FULL),
    (("Bachelor's", "CS"), ("B.S.", "Computer Science"), FULL),
    (("B.Tech", "CSE"), ("Bachelor of Technology", "Computer Science Engineering"), FULL),
    (("BA", "Economics, Minor: Statistics"), ("BA", "Statistics, Minor: Economics"), "notfull"),
    (("BA", "Economics; Minor: Statistics; Minor: Math"), ("BA", "Economics; Minor: Math; Minor: Statistics"), FULL),
    (("BSN", None), ("B.S.", "Nursing"), FULL),
    (("MBA", None), ("Master of Business Administration", None), FULL),
    (("High School Diploma", "College Prep"), ("High School Diploma", None), (2, 3, 2)),
    (("Semester 1", None), ("Semester 2", None), "notfull"),
    (("HSC", "Science"), ("HSC, Science", None), FULL),
    (("SSC", "Science"), ("SSC, Science", None), FULL),
    (("GED", None), ("GED", None), FULL),
    (("HND", "Engineering"), ("HND in Engineering", None), FULL),
    (("PGDM", "Marketing"), ("PGDM in Marketing", None), FULL),
    (("HSC", "Higher Secondary"), ("HSC", None), "?"),
    (("M.S.", "MBA"), ("M.S.", None), "notfull"),
    (("M.S., MBA", None), ("M.S.", "MBA"), FULL),
    (("B.A.", "B.S. in Economics"), ("B.A.", "Economics"), "notfull"),
    (
        ("Bachelor of Arts, Bachelor of Science in Economics", None),
        ("Bachelor of Arts", "Bachelor of Science in Economics"),
        FULL,
    ),
    (("Bachelor of Science", "Bachelor of Science in Nursing"), ("Bachelor of Science in Nursing", None), FULL),
    (("MBA", "Master of Business Administration"), ("MBA", None), FULL),
    (("B.S. in Biology, M.S. in Chemistry", None), ("B.S. in Chemistry, M.S. in Biology", None), "notfull"),
    (("Master's in Computer Science (MS)", None), ("M.S.", "Computer Science"), FULL),
    (("BS", "BA"), ("BS", "BA"), FULL),
    (("BS", None), ("BA", None), "notfull"),
    (("Executive MBA", None), ("MBA", None), "notfull"),
    (("BSCS", None), ("Bachelor of Science in Computer Science", None), FULL),
    (("Diploma in Nursing", None), ("Certificate in Nursing", None), FULL),
]


@pytest.mark.parametrize(("label", "pred", "expect"), [c for c in CASES if c[2] != "?"])
def test_degree(label, pred, expect):
    r = score_resume_v5(d(*label), d(*pred))["sections"]["education"]
    got = (r["correct"], r["expected"], r["predicted"])
    if expect == FULL:
        assert got[0] == got[1]
    elif expect == "notfull":
        assert got[0] < got[1]
    else:
        assert got == expect


@pytest.mark.parametrize(
    ("label", "pred", "expect"),
    [
        (("Master of Engineering (M.E.)",), ("Master of Engineering",), FULL),
        (("Master of Engineering (M.E.)",), ("Master of Engineering", "Mechanical Engineering"), (2, 2, 3)),
        (("Bachelor of Science", "Applied Mathematics"), ("Bachelor of Applied Science", "Mathematics"), "notfull"),
        (("Master of Science", "Management"), ("Master of Management Science",), "notfull"),
    ],
)
def test_restatement_and_type_words(label, pred, expect):
    test_degree(label, pred, expect)
