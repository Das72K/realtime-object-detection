import pytest

from rtod.geometry import point_in_polygon, segments_intersect, side_of_line

H_LINE = ((0, 50), (100, 50))  # horizontal, drawn left -> right


def test_side_of_line_image_coords():
    # y grows downwards: below the line is the positive (right-hand) side of left->right
    assert side_of_line(*H_LINE, (50, 80)) == 1
    assert side_of_line(*H_LINE, (50, 20)) == -1
    assert side_of_line(*H_LINE, (50, 50)) == 0


def test_side_flips_with_line_direction():
    assert side_of_line((100, 50), (0, 50), (50, 80)) == -1


def test_segments_cross():
    assert segments_intersect((50, 0), (50, 100), *H_LINE)


def test_segments_parallel_do_not_cross():
    assert not segments_intersect((0, 60), (100, 60), *H_LINE)


def test_segments_miss_beyond_endpoint():
    # crosses the infinite line y=50, but at x=150 -> outside the drawn segment
    assert not segments_intersect((150, 0), (150, 100), *H_LINE)


def test_segments_touching_endpoint_counts():
    assert segments_intersect((50, 50), (50, 100), *H_LINE)


def test_segments_collinear_overlap():
    assert segments_intersect((80, 50), (120, 50), *H_LINE)
    assert not segments_intersect((120, 50), (150, 50), *H_LINE)


SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
CONCAVE_U = [(0, 0), (3, 0), (3, 7), (7, 7), (7, 0), (10, 0), (10, 10), (0, 10)]


@pytest.mark.parametrize("pt,inside", [((5, 5), True), ((15, 5), False), ((-1, 5), False), ((5, 11), False)])
def test_point_in_square(pt, inside):
    assert point_in_polygon(pt, SQUARE) is inside


def test_point_in_concave_polygon():
    assert point_in_polygon((1, 2), CONCAVE_U)      # left arm of the U
    assert point_in_polygon((8, 2), CONCAVE_U)      # right arm
    assert not point_in_polygon((5, 2), CONCAVE_U)  # the notch between the arms
    assert point_in_polygon((5, 9), CONCAVE_U)      # bottom of the U
