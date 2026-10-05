import pandas as pd

from code.prepare_data import select_subset


def test_select_subset_filters_by_phenotype_and_columns():
    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3"],
            "phenotype": ["case", "control", "case"],
            "sex": ["female", "male", "female"],
            "gene_a": [1, 2, 3],
            "gene_b": [4, 5, 6],
        }
    )

    result = select_subset(
        df,
        phenotype="case",
        columns=["patient_id", "phenotype", "gene_a"],
    )

    expected = pd.DataFrame(
        {
            "patient_id": ["p1", "p3"],
            "phenotype": ["case", "case"],
            "gene_a": [1, 3],
        }
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)
