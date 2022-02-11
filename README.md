"# MulticraftAndMMLADataMining" 

# Survey 
1. Instructions on how to use Gaze Processing
    * Make a folder, put the video + corresponding gaze data in the folder, then run the script 
    * Can do this for multiple videos, just make multiple folders 
    * Output appears in directory of script 
2. Instructions on how to use Saccade Processing 
    * To run with real data 
        - Get data_output_w_times_combined.csv from Qualtrics 
            - Might need to add data (the file locations and whatnot)
        - Get corresponding data with it 
            - Bring over the videos in the right places or the original format, however it is reflected on the csv
    * To test
        - Make a folder with the appropriate path shown in the data_output_w_times file, put the video + corresponding gaze data in the folder, then run the script 
            - Alternatively, update the data_output_w_times file to whatever the path is for the video + gaze data
        - Can do this for multiple videos, just make multiple folders, and/or update the data_output_w_times file accordingly
        - Output appears in directory of script 

# Note: 
    * data_output_w_times_combined and qualtrics_mapping_combined are the results two studies combined
    * excluding_questions is the results combined, but only with the questions they have in commong 
    * exploration was the initial data that was run to see how the code worked on the current data  

# Notes on Saccade Processing 
* Reuse code potential 
    - 