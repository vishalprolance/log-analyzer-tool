import streamlit as st
import pandas as pd
import time, math
import logging, os, time
from datetime import datetime
import Chunking, VectorDB, LLM, SmallSummary, LargeSummary

##### Logging #####
log_dir = "syslogs/"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f"{__name__}.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

logger.info("Staring up...")
logger.debug("Debug information logged.")
logger.error("An error occurred.")

##### Logging #####
##### Header #####

st.title("Log analyzer for priority incidents 🧠")
st.write("---")

st.title(":blue[Initiate root cause analysis] :warning:")

col1, col2 = st.columns(2)
with col1:
    st.write('Select your LLM')
    option = st.selectbox(
        "Choose from dropdown",
        ("Meta", "Titan", "Mistral"),
    )

with col2:
    st.write('Select your application framework/s')
    options = st.multiselect(
        "Select all those frameworks applicable for your application",
        ["Dot Net", "IBM BPM", "Linux"],
    )

    #Create a dictionary to map framework names to their respective codes
    frameworks={
        'Dot Net':'dotnet',
        'IBM BPM':'ibmbpm',
        'Linux':'linux'
    }

workaround = st.text_area(
    "# Error Report: Action Taken (Optional)",
    )

st.write(f'{len(workaround)} characters.')

########## UPOLOAD FILE
if not os.path.exists('input'):
    os.makedirs('input')

uploaded_file = st.file_uploader(  "Upload your log file", 
                                    accept_multiple_files=False,
                                    type=['txt'],
                                    )
if uploaded_file is None:
    st.error("Please upload a log file for analysis.")
else:
    # Define file path for saving the uploaded file
    file_name = os.path.join('input', uploaded_file.name)

    # Save the uploaded file to the 'input' directory
    with open(file_name, 'wb') as file:
        file.write(uploaded_file.getbuffer())

    # Read and display the file in the text area
    with open(file_name, 'r') as file:
        st.text_area('Input Log File', file.read(), height=400)
        
if st.button("Analyze :mag:", type="primary"):
    
    start_time = time.time()
    
    progress_text = "Initiating server..."
    my_bar = st.progress(0, text=progress_text)
    time.sleep(1)
    
    ########## Backend - Chunking
    progress_text = "Chunking log file..."
    my_bar.progress(20, text=progress_text)

    # Assuming the file path is stored in 'file_path' after uploading
    file_name = os.path.join('input', uploaded_file.name)
    chunking_obj = Chunking.SemanticChunking(file_name)  # Initialize with file path if needed
    all_chunks = chunking_obj.getChunks(file_name)  # Pass the file path or file name to getChunks()

    ######### Backend - VectorDB
    progress_text = "Initiaing vectorDB and inserting vectors..."
    my_bar.progress(30, text=progress_text)
    # Check if any frameworks are selected
    if options:
        # Calculate progress step
        progress_step = 20 / len(options)  # We have 20% (50 - 30) to work with
        current_progress = 31
        
    vectorDBs = {}
    for framework in options:
        progress_text = f"Initiating vectorDB for {framework}..."
        my_bar.progress(int(current_progress), text=progress_text)
        
        framework_code = frameworks[framework]
        vectorDB_obj = VectorDB.Milvus_VDB(framework_code)
        vectorDBs[framework] = vectorDB_obj.getVectorDB()
        
        current_progress += progress_step
        my_bar.progress(min(int(current_progress), 50))  # Ensure we don't exceed 50

    # Final progress update
    my_bar.progress(49, text="All VectorDBs initialized successfully!")
    
    ######### Backend - LLM
    progress_text = "Initiaing LLM API..."
    my_bar.progress(50, text=progress_text)
    llm_obj = LLM.Anthropic()
    llm_small = llm_obj.get('light', 500)
    llm_large = llm_obj.get('heavy', 20000)
    
    # ######### Backend - Chunk Summaries
    progress_text = "Summerizing chunks..."
    my_bar.progress(70, text=progress_text)
    chunk_summary_obj = SmallSummary.ChunkSummaries()
    fileOfSummaries = chunk_summary_obj.getSummeries(all_chunks, llm_small, vectorDB, workaround, frameworks[option])
    
    ######## Backend - Report Generation
    progress_text = "Generating report..."
    my_bar.progress(80, text=progress_text)
    report_generation_obj = LargeSummary.GenerateReport()
    total_trial=5
    while total_trial > 0:
        try:
            report=report_generation_obj.getSummary(fileOfSummaries, llm_large)
            break
        except Exception as e:
            print("Failed report generation", e)
            total_trial -= 1  
            
    ######## Printing the report
    progress_text = "Publishing the report..."
    my_bar.progress(100, text=progress_text)
    time.sleep(2)
    
    my_bar.empty()
    
    # report={'Cause': 'The incidents were caused by:\n\n- Missing view file resulting in ViewNotFoundException\n- Invalid file path leading to FileNotFoundException \n- Database connectivity issues causing SqlException\n- Attempt to divide by zero resulting in DivideByZeroException\n- Potential null reference or divide by zero in StatsService', 'Solution': 'Some solutions to fix these are:\n\n- For missing view, ensure view file exists in expected location\n- Validate file path before processing file import\n- Check database connection string, network connectivity\n- Add null check before division to prevent divide by zero\n- Validate input arrays to avoid null reference\n- Add detailed exception handling and logging\n- Handle expected exceptions like FileNotFound and DivideByZero specifically\n- Review calculation logic to prevent errors\n- Add more logging at exception origin to identify root cause'}
    
    ####### Display Report
    st.title(":blue[Incident Analysis Report] :zap:")
    
    tab1, tab2, tab3 = st.tabs(["Analysis", "Solution", "References"])

    with tab1:
        st.header("Incident Cause Analysis")
        md = st.text_area('Below is the cause report generated', report['Cause'], height=400)
    
    with tab2:
        st.header("Suggested Solution")
        md = st.text_area('Below is the solution report generated', report['Solution'], height=400)
    
    with tab3:
        with open('output/interim_output.txt', 'r') as ref_file:
            st.header("Chunk Summaries Used for Report")
            md = st.text_area('Below are the references utilized', ''.join(ref_file), height=400)
    
    ####### Save Report
    with open('output/final_report.txt', 'w') as final_report:
        final_report.write('---------------- CAUSE ANALYSIS ----------------\n')
        final_report.write(report['Cause'])
        final_report.write('\n\n---------------- SUGGESTED SOLUTION ----------------\n')
        final_report.write(report['Solution'])
        final_report.write('\n\n---------------- REFERENCES ----------------\n')
        with open('output/interim_output.txt', 'r') as ref:
            final_report.write(''.join(ref))
    
    end_time = time.time()
    
    st.download_button("Download Report", open('output/final_report.txt'))
    
    
    ####### Calculate Impact
    with open('output/interim_output.txt', 'r') as ref_file:
        token_count = 0
        file_size = 0
        for line in ref_file:
            token_count += 2 * llm_large.get_num_tokens(line)
            file_size += 1
        print(file_size, token_count)
        st.write(f"""
        ## :blue[Impact] 🚀
        - Time taken by *GenAI* method: **{math.ceil((end_time-start_time)/60)} mins**
        - Cost incurred by *GenAI* method: **${round((token_count//1000)*0.024*2, 4)}**
        - It would have taken **{file_size//120} hours** and costed **${(file_size//120)*30}** to analyze it manually.
        """)



