INSERT OVERWRITE DIRECTORY 'ValentinMSU17_hiveout'
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
STORED AS textfile
SELECT *
FROM
    hw2_pred;