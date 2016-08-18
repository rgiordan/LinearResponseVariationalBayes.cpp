# include "gtest/gtest.h"
# include "box_parameters.h"
# include <string>

# define DOUBLE_INF std::numeric_limits<double>::infinity()

void TestParams(double x, double x_lower, double x_upper, double scale,
                std::string msg) {

    double x_box = box_parameter(x, x_lower, x_upper, scale);
    EXPECT_TRUE(x_box > x_lower) << msg;
    EXPECT_TRUE(x_box < x_upper) << msg;

    double x_unbox = unbox_parameter(x_box, x_lower, x_upper, scale);
    EXPECT_NEAR(x_unbox, x, 1e-12) << msg;
};

TEST(BoxParameters, basic) {
    TestParams(3.0, -1.0, 2.3, 1.0, "Negative bound, unit scale");
    TestParams(3.0, 1.1, 2.3, 1.0, "Positive bound, unit scale");
    TestParams(3.0, -1.0, 2.3, 1.7, "Negative bound, non-unit scale");
    TestParams(3.0, -1.0, DOUBLE_INF, 1.0, "Infinite bound, unit scale");
    TestParams(3.0, -1.0, DOUBLE_INF, 1.7, "Infinite bound, non-unit scale");
};
