add archive projects/2/filter_predict.py;
add archive 2.joblib;
add archive projects/2/model.py;
add archive projects/2/train.py;
add archive projects/2/train.sh;

INSERT INTO TABLE hw2_pred
SELECT
    TRANSFORM(*) USING 'filter_predict.py' AS (id, prediction)
FROM
    hw2_test;