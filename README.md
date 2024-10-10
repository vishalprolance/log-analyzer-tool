# log-analyzer-tool
Log analyzer tool using GenAI

#Create python virtual environment
python -m venv dev

#Activate virtual environment
source dev/bin/activate

#Deactivate virtual environment
deactivate

#Install requirement from requirement.txt file
pip install -r requirement.txt

--------------------------------------------------app.py----------------------------------------------------------------------------
This Streamlit app is designed to perform log analysis using Generative AI techniques. The app provides a graphical interface for users to upload log files, analyze them with a Large Language Model (LLM), and generate a report. Below is a detailed breakdown of the code and its functionality.

1. Import Statements
The code imports various libraries:
    Streamlit (st): For building the interactive web app.
    Pandas (pd): Although imported, it's not used in this snippet but commonly used for data manipulation.
    time, math: For time-related functions and mathematical operations.
    logging, os: For logging system information and file operations.
    datetime: To handle date and time values.
    Custom modules (Chunking, VectorDB, LLM, SmallSummary, LargeSummary) are imported, which handle specific backend processes related to log analysis.

2. Logging Initialization
The code sets up logging:
    Logger Setup: Initiates a logger with debug-level logging.
    File Handler: Logs are stored in a file within a syslogs/ directory.
    python code:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(f"syslogs/{__name__}.log")
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.info("Starting up...\n")

3. App Title and User Input (Streamlit Interface)
    Title and Description: The app displays a title (Log analyzer 🧠) and a section where users can choose their desired LLM model and application framework.
    LLM Selection: Users can select from a dropdown list of LLMs such as "Anthropic", "Meta", "Titan", etc.
    Framework Selection: A multi-select dropdown allows users to choose the applicable frameworks for their application (e.g., ".NET", "IBM BPM", "Linux").
    Text Area: Users can optionally provide instructions or workarounds.
    
    python code:
        col1, col2 = st.columns(2)
        with col1:
            option = st.selectbox("Choose from dropdown", ("Antrhopic", "Meta", "Titan", "Cohere", "Jurassic", "Mistral"))

        with col2:
            options = st.multiselect("Select all those frameworks applicable", ["Dot Net", "IBM BPM", "Linux"])

4. File Upload
The app allows users to upload a log file for analysis. The uploaded file is saved to the input/user_input.txt file and displayed in a text area.

    python code:
        uploaded_file = st.file_uploader("Upload your log file", type=['txt'])
        if uploaded_file is not None:
            with open('input/user_input.txt', 'wb') as file:
                file.write(uploaded_file.read())
            with open('input/user_input.txt', 'r') as file:
                st.text_area('Input Log File', ''.join(file), height=400)

5. Analysis Button and Progress Bar
When the user clicks the "Analysis" button, the following backend processes are initiated:
    Progress Bar: A progress bar shows the status of different stages of the analysis.
    Chunking: The log file is broken into chunks using the Chunking module.
    VectorDB: The VectorDB module inserts vector representations of the log chunks into a vector database.
    LLM API: The LLM API is initialized to perform analysis on the log chunks.
    Summaries: Small and large summaries are generated using the SmallSummary and LargeSummary modules.
    
    python code:
        if st.button("Analysis :mag:", type="primary"):
            # Progress bar logic and backend processes go here

6. Backend Process Descriptions
    Chunking: The log file is divided into manageable chunks.
    VectorDB: Vector representations of these chunks are stored in a vector database (Chroma).
    LLM API: The selected LLM performs light and heavy analyses on the chunks.
    Summarization: Small summaries are generated for each chunk, and a large summary is compiled into a final report.

7. Report Generation
A report is generated based on the analysis, and the app displays the following sections:
    Analysis: Shows the causes of the issues found in the logs.
    Solution: Displays recommended solutions.
    References: Contains chunk summaries used in generating the report.
    python code:
        st.title(":blue[Incident analysis report] :zap:")
        tab1, tab2, tab3 = st.tabs(["Analysis", "Solution", "References"])
        # Display analysis, solution, and references

8. Report Download
The final report is saved as a text file (final_report.txt) and can be downloaded by the user through a Download button.

    python code:
        st.download_button("Download Report", open('output/final_report.txt'))

9. Cost and Time Impact
The app calculates the time taken and cost incurred by the Generative AI method and compares it with manual analysis estimates. This data is displayed to the user.

    python code:
        st.write(f"""
        ## :blue[Impact] 🚀
        - Time taken: **{math.ceil((end_time-start_time)/60)} mins**
        - Cost: **${round((token_count//1000)*0.024*2, 4)}**
        - Estimated manual analysis time: **{file_size//120} hours** and cost: **${(file_size//120)*30}**
        """)

10. Conclusion
This Streamlit app provides an end-to-end solution for analyzing log files using LLMs, summarizing the log content, and generating a detailed incident report. Users can interact with the app through a graphical interface, upload log files, and view the results in a well-organized report format. The app also offers time and cost insights, giving a clear comparison between AI-powered and manual log analysis.

----------------------------------------------------------Chunking.py-----------------------------------------------------------
This code defines an abstract Chunking class and two concrete implementations—LineBasedChunk and ErrorBasedChunk. The purpose of this module is to break (or "chunk") a log file or data into smaller segments, which can be processed later. Chunking is helpful in scenarios where logs or data are too large to handle as a whole.

1. Abstract Class Chunking
The base class Chunking is an abstract class using Python's ABC module. This class provides a blueprint for chunking operations but doesn't implement any logic itself. The concrete classes that inherit from Chunking must implement the getChunks method.
    Abstract Method (getChunks):
    This method is abstract, meaning it must be implemented by any subclass. It takes a file name as input and is expected to return a list of strings, where each string represents a chunk from the file.
    python code:
        class Chunking(ABC):
            @abstractmethod
            def getChunks(self, file_name: str) -> list[str]:
                """
                Chunks the file by a certain number of lines.
                """
                pass

2. Line-Based Chunking (LineBasedChunk)
This class inherits from Chunking and implements the chunking logic based on the number of lines. The file is divided into chunks, each containing a fixed number of lines.
    Constructor (__init__):
    The constructor initializes the object with the file_name and creates the file path (input/{file_name}).

    getChunks Method:
    This method reads the file and divides it into chunks, where each chunk contains 10 lines (or fewer for the last chunk). It iterates over the file line by line, concatenating lines into a chunk, and when the chunk reaches 10 lines, it is stored in all_chunks.

    Print Statements:
    The method prints messages to indicate when chunking starts and ends, as well as the total number of chunks extracted.

    python code:
        class LineBasedChunk(Chunking):
            def __init__(self, file_name):
                self.path=f'input/{file_name}'
                
            def getChunks(self):
                print(f'\n===== CHUNKING LOG FILE =====\n')
                lines_per_chunk = 10
                all_chunks = []

                with open(self.path) as bigfile:
                    chunk=""
                    for lineno, line in enumerate(bigfile):
                        if (lineno+1) % lines_per_chunk == 0:
                            all_chunks.append(chunk)
                            chunk=""
                        chunk += '\n' + line

                    all_chunks.append(chunk)  # Append the last chunk

                print(f'Total {len(all_chunks)} chunks extracted.\n===== END OF LOG CHUNKING =====\n')
                return all_chunks

3. Error-Based Chunking (ErrorBasedChunk)
This class chunks the data based on an error threshold rather than a fixed number of lines. It is designed to chunk data dynamically, depending on some calculated error value.
    Constructor (__init__):
    Takes the data to be chunked and an error threshold value. This threshold determines when to break the current chunk and start a new one.

    getChunks Method:
    Iterates through the data, adding items to the current chunk and recalculating the error. If the error exceeds the threshold, the chunk is split, and a new chunk is started.

    Error Calculation (calculate_error):
    This method is a placeholder for calculating the error in the chunk. The actual logic for error calculation must be defined depending on the specific use case.

    python code
        class ErrorBasedChunk(Chunking):
            def __init__(self, data, error_threshold):
                super().__init__(data)
                self.error_threshold = error_threshold

            def getChunks(self, data, error_threshold):
                """
                Chunks the data based on an error threshold.
                """
                all_chunks = []
                current_chunks = []
                current_error = 0

                for item in self.data:
                    current_chunk.append(item)
                    current_error = self.calculate_error(current_chunk)

                    if current_error > self.error_threshold:
                        all_chunks.append(current_chunks[:-1])  # Exclude the last item
                        current_chunks = [item]  # Start a new chunk with the last item
                        current_error = 0
                    
                    if current_chunks:
                        all_chunks.append(current_chunks)

                print(f'Total {len(all_chunks)} chunks extracted.\n===== END OF LOG CHUNKING =====\n')
                return all_chunks

            def calculate_error(self, chunk):
                """
                Placeholder for error calculation logic.
                """
                return 0  # Replace with actual error calculation

Summary:
Chunking Abstract Class: Defines the blueprint for any chunking operation.
LineBasedChunk Class: Implements chunking by a fixed number of lines (10 lines per chunk). This method is useful when logs are too large to process at once.
ErrorBasedChunk Class: Implements chunking based on a dynamically calculated error threshold. This method is more sophisticated and allows the user to split data based on the severity or error in the data. The calculate_error method needs to be customized for specific error-calculation logic.
This design follows the Strategy Pattern, where the abstract class defines the method signature, and different concrete classes implement the chunking strategy either by lines or error.

-------------------------------------------------------VectorDB.py----------------------------------------------------
This code defines an abstract class VectorDB and two concrete implementations: FAISS_VDB and Chroma_VDB. The classes are responsible for handling vector databases, particularly for embedding and storing vectorized data from PDF manuals using Amazon Bedrock's embeddings and various libraries from LangChain.

1. VectorDB (Abstract Base Class)
This is an abstract base class that defines a contract for vector database operations. The class includes:
getVectorDB(self, path: str) -> object: This is an abstract method that needs to be implemented by any subclass. It is responsible for returning an instance of the vector database.

2. FAISS_VDB (Concrete Class for FAISS Vector Database)
This class implements the VectorDB interface using FAISS as the underlying vector database.

    __init__(self, directory):

The constructor initializes the directory where PDF manuals are stored.
Example directory path: 'data/dotnet' for a DotNet server.
    getVectorDB(self):

    Imports: Necessary libraries for vector storage, PDF loading, document splitting, and embeddings are imported from langchain and utils.bedrock.
    Bedrock Client: Connects to Amazon Bedrock to generate embeddings using the amazon.titan-embed-text-v1 model.
    Document Loading: PDF manuals are loaded from the specified directory using PyPDFDirectoryLoader.
    Document Splitting: Splits loaded documents into chunks of size 1000 characters with an overlap of 100 characters using RecursiveCharacterTextSplitter.
    Embedding Generation: For each document chunk, embeddings are generated using the BedrockEmbeddings API.
    Embedding Error Handling: In case of an AccessDeniedException, the error message is printed with links to AWS IAM documentation.
    Vector Insertion: The document embeddings are inserted into the FAISS vector store.
    Return: The FAISS vector database is returned for further use.

3. Chroma_VDB (Concrete Class for Chroma Vector Database)
This class provides a similar implementation as FAISS_VDB but uses Chroma as the underlying vector database.

    __init__(self, directory):

Similar to FAISS_VDB, initializes the directory where the PDF manuals are stored.
    getVectorDB(self):

    Imports: Imports the same libraries as FAISS_VDB for document loading, splitting, and embedding generation.
    Document Loading: Loads PDF files from the specified directory.
    Document Splitting: Documents are split into chunks of 1000 characters with 100 characters overlap.
    Embedding Generation: Embeddings are generated using BedrockEmbeddings.
    Error Handling: Handles any embedding-related exceptions, specifically catching AccessDeniedException and providing AWS IAM troubleshooting links.
    Vector Insertion: The embeddings are inserted into a Chroma vector database.
    Return: The Chroma vector database is returned for further use.
    Key Features of Both Classes
        Directory Management: Both classes accept a directory path where PDF files are stored.
        Document Loading and Splitting: PDF manuals are loaded, and the text is split into manageable chunks for embedding.
        Amazon Bedrock Embeddings: Both classes use Amazon Bedrock to generate embeddings using a pre-trained model (amazon.titan-embed-text-v1).
    Vector Store Implementation:
        FAISS: Efficient similarity search and vector insertion are handled by the FAISS library.
        Chroma: Provides another vector store option for storing and querying vectors.
    Error Handling
    Embedding Generation: Both classes include error handling for generating embeddings. If there is an access issue with Bedrock, the program prints an error message with links to AWS troubleshooting guides.
Conclusion
The VectorDB abstract class defines the interface for working with vector databases.
FAISS_VDB and Chroma_VDB are concrete implementations that load PDF documents, split them into chunks, generate embeddings via Amazon Bedrock, and store them in either FAISS or Chroma vector databases.

--------------------------------------------------------LLM.py-------------------------------------------------------------
This code defines an abstract class LLM and several concrete classes (Anthropic, Meta, Titan, Cohere, and Mistral) that implement different large language models (LLMs). The purpose of these classes is to interact with Amazon Bedrock services and provide the functionality to return specific LLMs based on the user’s requirements (e.g., model type or size).

1. LLM (Abstract Base Class)
This is the abstract base class that defines a contract for different types of LLMs.
    get(self, typ) -> object:
This is an abstract method that must be implemented by any subclass.
    Parameters:
    typ: Can be either 'light' or 'heavy', specifying whether to use a lighter or heavier (larger) version of the model.
    Return: This method returns an LLM object.

2. Anthropic (Concrete Class for Anthropic LLMs)
    Purpose: This class provides two models from Anthropic’s LLMs:
    claude-instant-v1 (light version)
    claude-v2 (heavy version)
    get(self, typ='light', max_token=500):
    Retrieves an Anthropic model using Amazon Bedrock.
    Parameters:
    typ: 'light' or 'heavy'. The default is 'light'.
    max_token: Maximum number of tokens to sample (default: 500).
    Bedrock Client: Uses bedrock.get_bedrock_client to retrieve a client for Amazon Bedrock.
    Model Selection: Depending on the value of typ, either claude-instant-v1 or claude-v2 is selected.
    Return: A Bedrock object representing the Anthropic model is returned.

3. Meta (Concrete Class for Meta LLMs)
    Purpose: Provides access to Meta’s LLaMA 2 model (llama2-13b-chat-v1) using Amazon Bedrock.

    get(self, typ='light'):

    Retrieves the Meta LLaMA 2 model with max_tokens_to_sample set to 200.
    Bedrock Client: Initializes a client for Amazon Bedrock using bedrock.get_bedrock_client.
    Return: A Bedrock object representing the Meta LLaMA 2 model.

4. Titan (Concrete Class for Amazon Titan LLMs)
    Purpose: Provides access to Amazon’s Titan models:

    titan-text-lite-v1 (light version)
    titan-text-express-v1 (heavy version)
    get(self, typ='light'):

    Retrieves either the light or heavy version of the Amazon Titan model.
    Model Selection: Based on the typ parameter, either titan-text-lite-v1 or titan-text-express-v1 is selected.
    Return: A Bedrock object representing the Amazon Titan model.

5. Cohere (Concrete Class for Cohere LLMs)
    Purpose: Provides access to Cohere’s models:
    command-light-text-v14 (light version)
    command-text-v14 (heavy version)
    get(self, typ='light', max_token=500):
    Retrieves a Cohere model with either the light or heavy variant.
    Model Selection: Based on the typ parameter, either command-light-text-v14 or command-text-v14 is selected.
    Return: A Bedrock object representing the Cohere model.

6. Mistral (Concrete Class for Mistral LLMs)
    Purpose: Provides access to Mistral’s models:

        mistral-small-2402-v1:0 (light version)
        mistral-large-2402-v1:0 (heavy version)
        get(self, typ='light', max_token=500):

    Retrieves a Mistral model based on the specified type.
    Model Selection: Depending on the value of typ, either mistral-small-2402-v1:0 or mistral-large-2402-v1:0 is selected.
    Return: A Bedrock object representing the Mistral model.
    Key Features Across All Classes:
    Type Selection: Each LLM class allows selection between a lighter model (typ='light') or a heavier model (typ='heavy').
    Amazon Bedrock: All the classes use Amazon Bedrock to manage the LLMs and generate responses. The Bedrock client is retrieved using the bedrock.get_bedrock_client utility function.
    Error Handling: For each class, there is basic error handling for invalid types (typ not being 'light' or 'heavy').
    Token Configuration: Most models allow customization of the maximum number of tokens to sample during generation, usually set to 200 or 500 by default.
Conclusion
The LLM abstract class defines a unified interface for accessing various large language models through Amazon Bedrock. The concrete classes—Anthropic, Meta, Titan, Cohere, and Mistral—provide access to specific models offered by these organizations. Each class supports customizable configurations like choosing between light or heavy model types and setting token limits for generation.

----------------------------------------------------------