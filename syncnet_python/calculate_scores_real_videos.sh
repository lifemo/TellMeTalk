rm all_scores.txt
yourfilenames=`D:\study\py\wav2lip\dataset_porcessed\vgg_out\video`

for eachfile in yourfilenames
do
   python run_pipeline.py --videofile $1/$eachfile --reference wav2lip --data_dir tmp_dir
   python calculate_scores_real_videos.py --videofile $1/$eachfile --reference wav2lip --data_dir tmp_dir >> all_scores.txt
done
