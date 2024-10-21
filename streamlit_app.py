import streamlit as st
import streamlit.components.v1 as components
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, Document, VectorStoreIndex
from llama_index.llms.anthropic import Anthropic
from anthropic import Anthropic as AnthropicClient, HUMAN_PROMPT, AI_PROMPT
import uuid

# EMBEDDINGS
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

embed_model = OpenAIEmbedding(
    model="text-embedding-3-large",
    dimensions=1536,
)

# Load environment variables
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

# Initialize Pinecone
index_name = "llamaindex-docs"
if not PINECONE_API_KEY:
    st.error("Pinecone API key is not set. Please check your .env file.")
    st.stop()

# Streamlit app title
st.title("LlamaIndex Workflow with Anthropic, Mermaid, and Pinecone")

# Initialize components
@st.cache_resource
def init_components():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)
    
    # Initialize Anthropic LLM for text generation
    llm = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Setup the vector store and storage context
    vector_store = PineconeVectorStore(pinecone_index=index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Initialize Anthropic client
    anthropic_client = AnthropicClient(api_key=ANTHROPIC_API_KEY)
    
    return pc, index, llm, vector_store, storage_context, anthropic_client

# Call the initialization function
pc, index, llm, vector_store, storage_context, anthropic_client = init_components()

# Streamlit UI for user input
query = st.text_area("Enter your project idea:", "A mobile app for tracking personal carbon footprint", height=200)

# Custom Mermaid rendering function
def render_mermaid(code: str) -> None:
    components.html(
        f"""
        <pre class="mermaid">
            {code}
        </pre>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """,
        height=500,  # Adjust the height as needed
    )

# Define the workflow function
def main_workflow(query):
    similar_projects = find_similar_projects(query)
    st.subheader("Similar Project Ideas")
    st.write(similar_projects)

    project_brief = brainstorm_step(query)
    st.subheader("Project Design Brief")
    st.write(project_brief)

    flowchart = flowchart_step(project_brief)
    st.subheader("Flowchart and Recommendations")
    st.write(flowchart)

    # Extract and render Mermaid diagram
    mermaid_start = flowchart.find("```mermaid")
    mermaid_end = flowchart.find("```", mermaid_start + 10)
    if mermaid_start != -1 and mermaid_end != -1:
        mermaid_code = flowchart[mermaid_start+10:mermaid_end].strip()
        render_mermaid(mermaid_code)
        # STEP 3 is conditional on mermaid running
        user = user_step(project_brief)
        st.subheader("Persona, Scenario, and User Interview Questions")
        st.write(user)
    else:
        st.error("Mermaid flowchart not found in the generated response.")

    return flowchart

# New function to find similar projects
def find_similar_projects(query):
    # Create a temporary document from the query
    temp_doc = Document(text=query)
    temp_index = VectorStoreIndex.from_documents([temp_doc], storage_context=storage_context)
    
    # Perform a similarity search
    retriever = temp_index.as_retriever(similarity_top_k=3)
    similar_docs = retriever.retrieve(query)
    
    if similar_docs:
        return "\n\n".join([f"Similar Project: {doc.text}" for doc in similar_docs])
    else:
        return "No similar projects found."

# Workflow steps
# STEP 1
def brainstorm_step(query):
    brainstorm_prompt = f"{HUMAN_PROMPT} Use '{query}' as the problem and define the solution by outlining experience highlighting pain points and explaining how your solutions resolve an issue, conflict or problem. Create and output a Project Design Brief with the following sections:\n\n1. Target Market\n2. Target Audience\n3. Competitors\n4. Project Description\n5. Technical Requirements\n6. Expected Outcome from using the product\n7. Estimated number of potential users\n8. Estimated number of potential business partners\n9. Expected revenue for first year in operation\n10. Explanation of monetization strategy\n\nPlease format your response as a structured document with clear headings for each section.{AI_PROMPT}"

    try:
        # Generate a response using Anthropic LLM
        response = anthropic_client.completions.create(
            model="claude-2",
            prompt=brainstorm_prompt,
            max_tokens_to_sample=100
        )
        project_design_brief = response.completion
        
        # Store the project brief in Pinecone
        doc_id = str(uuid.uuid4())
        doc = Document(text=project_design_brief, id_=doc_id)
        VectorStoreIndex.from_documents([doc], storage_context=storage_context)
        
        return project_design_brief
        
    except Exception as e:
        st.error(f"An error occurred while generating the project brief: {str(e)}")
        return None

# STEP 2
def flowchart_step(project_design_brief):
    flowchart_prompt = f"{HUMAN_PROMPT} Based on the following Project Design Brief, please:\n\n1. Create a Mermaid flowchart describing the basic architecture of the project.\n2. Provide recommendations or suggestions on other features or considerations that might be useful.\n\nProject Design Brief:\n{project_design_brief}\n\nPlease format your response in two sections:\n1. Mermaid Flowchart\n2. Recommendations and Suggestions\n\nFor the Mermaid flowchart, use the following syntax:\n```mermaid\ngraph TD\n    A[Start] --> B[Process]\n    B --> C[End]\n```\n\nReplace the example with an appropriate flowchart for the project.{AI_PROMPT}"

    try:
        # Generate flowchart response
        response = anthropic_client.completions.create(
            model="claude-2",
            prompt=flowchart_prompt,
            max_tokens_to_sample=100
        )
        return response.completion
    except Exception as e:
        st.error(f"An error occurred while generating the flowchart: {str(e)}")
        return None

# STEP 3 - USER STORY
def user_step(project_design_brief):
    interview_prompt = f"{HUMAN_PROMPT} Based on the following Project Design Brief, please:\n\n1. Create a Persona based on the defined target audience and target market.\n\n2. Create a day in the life scenario for that Persona to describe the problem the application will solve highlighting the pain points of the experience.\n\n3. Create a list of questions for a user interview for the persona of a boat owner that services their high-end boat at a marina. Ask these questions to strategically balance both quantitative and qualitative aspects of user research principles.\n\nProject Design Brief:\n{project_design_brief}\n\nPlease format your response in three sections:\n1. Persona\n2. Scenario\n3. Interview\n\n{AI_PROMPT}"

    try:
        # Generate response
        response = anthropic_client.completions.create(
            model="claude-2",
            prompt=interview_prompt,
            max_tokens_to_sample=100
        )
        return response.completion
    except Exception as e:
        st.error(f"An error occurred while generating the persona, scenario, and user interview questions: {str(e)}")
        return None
        
# Run workflow on button click
if st.button("Run Workflow"):
    with st.spinner("Running workflow..."):
        result = main_workflow(query)
    if result:
        st.success("Workflow completed successfully!")
    else:
        st.error("Workflow failed to complete. Please check the error messages above.")
        
# Run workflow on button click
if st.button("Run Pinecone Index Stats"):
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("llamaindex-docs")
        stats = index.describe_index_stats()
        st.json(stats)
    except Exception as e:
        st.error(f"Error fetching Pinecone index stats: {str(e)}")
