add archive filter_predict.py;

INSERT INTO TABLE ValentinMSU17.hw2_pred
SELECT
    TRANSFORM(*) USING 'filter_predict.py' AS (id, prediction)
FROM
    ValentinMSU17.hw2_test;