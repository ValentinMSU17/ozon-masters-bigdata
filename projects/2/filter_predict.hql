add archive projects/2/filter_predict.py;
add archive 2.joblib;
add archive projects/2/model.py;

INSERT INTO TABLE ValentinMSU17.hw2_pred
SELECT
    TRANSFORM(*) USING 'filter_predict.py' AS (id, prediction)
FROM
    ValentinMSU17.hw2_test;