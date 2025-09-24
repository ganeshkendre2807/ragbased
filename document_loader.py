import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from datetime import datetime

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Text Q&A Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize the LLM model
@st.cache_resource
def init_model():
    """Initialize the ChatGoogleGenerativeAI model"""
    return ChatGoogleGenerativeAI(
        temperature=0.7,  # Reduced for more consistent answers
        model="models/gemini-1.5-flash-latest"
    )


# Initialize session state variables
def init_session_state():
    """Initialize session state variables"""
    if 'user_text' not in st.session_state:
        st.session_state.user_text = ""
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    if 'current_answer' not in st.session_state:
        st.session_state.current_answer = ""
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'file_name' not in st.session_state:
        st.session_state.file_name = ""


# Function to generate answer using LLM
def generate_answer(context_text, question):
    """Generate answer using the LLM with context and question"""
    try:
        # Initialize model and components
        model = init_model()

        # Create a more sophisticated prompt template
        prompt = PromptTemplate(
            template="""Based on the following text context, please answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer: Please provide a clear and informative answer based only on the information provided in the context. If the answer cannot be found in the context, please state that clearly.""",
            input_variables=['context', 'question']
        )

        parser = StrOutputParser()

        # Create the chain
        chain = prompt | model | parser

        # Generate answer
        result = chain.invoke({
            'context': context_text,
            'question': question
        })

        return result

    except Exception as e:
        return f"Error generating answer: {str(e)}"


# Main app function
def main():
    # Initialize session state
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.title("🤖 Text Q&A Assistant")
        st.markdown("""
        **How to use:**
        1. Paste your text or upload a .txt file
        2. Ask questions about your text
        3. Get AI-powered answers instantly

        **Features:**
        - Dynamic question answering
        - File upload support
        - Question history
        - Clean, responsive design
        """)

        # Clear button in sidebar
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state.user_text = ""
            st.session_state.qa_history = []
            st.session_state.current_answer = ""
            st.session_state.file_uploaded = False
            st.session_state.file_name = ""
            st.rerun()

    # Main content area
    st.title("📝 Ask Questions from Your Text")
    st.markdown("---")

    # Upload file section
    st.subheader("1. Upload a Text File or Enter Text Manually")
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

    if uploaded_file is not None:
        # Read file content
        file_content = uploaded_file.read().decode("utf-8")
        st.session_state.user_text = file_content
        st.session_state.file_uploaded = True
        st.session_state.file_name = uploaded_file.name
        st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

    # Top Text Input Box
    user_text = st.text_area(
        label="Or paste/type your text here:",
        value=st.session_state.user_text,
        height=200,
        placeholder="Enter the text you want to ask questions about...",
        help="This text will be used as context for answering your questions"
    )

    # Update session state when text changes
    if user_text != st.session_state.user_text:
        st.session_state.user_text = user_text
        st.session_state.file_uploaded = False
        st.session_state.file_name = ""

    # Show text statistics if text is provided
    if st.session_state.user_text:
        st.markdown("### 📊 Text Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Characters", f"{len(st.session_state.user_text):,}")
        with col2:
            st.metric("Words", f"{len(st.session_state.user_text.split()):,}")
        with col3:
            st.metric("Lines", f"{len(st.session_state.user_text.split(chr(10))):,}")
        with col4:
            # Estimate reading time (average 200 words per minute)
            word_count = len(st.session_state.user_text.split())
            reading_time = max(1, round(word_count / 200))
            st.metric("Est. Reading Time", f"{reading_time} min")

        # Show file info if uploaded from file
        if st.session_state.file_uploaded:
            st.info(f"📁 Content loaded from: **{st.session_state.file_name}**")

        # Warning for very large texts
        if len(st.session_state.user_text) > 50000:
            st.warning("⚠️ Large text detected! The AI will process the first portion for better performance.")

    st.markdown("---")

    # Question Input Section
    st.subheader("2. Ask Your Question")

    # Only show question input if user has provided text
    if st.session_state.user_text.strip():
        with st.form("question_form", clear_on_submit=True):
            question = st.text_input(
                label="What would you like to know about your text?",
                placeholder="e.g., What is the main topic? Summarize the key points...",
                help="Ask any question related to the text you provided above"
            )

            submitted = st.form_submit_button("🚀 Get Answer", type="primary")

            if submitted and question.strip():
                # Show loading spinner while processing
                with st.spinner("Generating answer..."):
                    answer = generate_answer(st.session_state.user_text, question)

                    # Add to history
                    st.session_state.qa_history.append({
                        'question': question,
                        'answer': answer,
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })

                    st.session_state.current_answer = answer

                st.rerun()
    else:
        st.info("👆 Please enter some text or upload a file above to start asking questions!")

    # Answer Display Section
    if st.session_state.current_answer:
        st.markdown("---")
        st.subheader("3. Latest Answer")

        # Display current answer in a styled container
        with st.container():
            st.markdown(f"""
            <div style="
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #4CAF50;
                margin: 10px 0;
            ">
                <h4 style="color: #2E8B57; margin-top: 0;">💡 Answer:</h4>
                <p style="margin-bottom: 0; line-height: 1.6;">{st.session_state.current_answer}</p>
            </div>
            """, unsafe_allow_html=True)

    # Question History Section
    if st.session_state.qa_history:
        st.markdown("---")
        st.subheader("4. Question History")

        # Show history in reverse order (newest first)
        for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):  # Show last 5
            with st.expander(f"Q{len(st.session_state.qa_history) - i}: {qa['question'][:50]}... ({qa['timestamp']})"):
                st.write("**Question:**")
                st.write(qa['question'])
                st.write("**Answer:**")
                st.write(qa['answer'])

    # Original Text Reference Section
    if st.session_state.user_text:
        st.markdown("---")
        with st.expander("📄 View Original Text"):
            st.text_area("Your original text:", value=st.session_state.user_text, height=150, disabled=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888; font-size: 14px;'>"
        "🤖 Powered by Google Gemini AI | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()



# simple document loader no ui ------------------------------------------------------------------------------
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import  StrOutputParser
# from langchain_community.document_loaders import  TextLoader
# load_dotenv()
#
# model = ChatGoogleGenerativeAI(
#     temperature=0.8,
#     model="models/gemini-1.5-flash-latest"
# )
#
# prompt = PromptTemplate(
#     template="ipl added to which popularity \n {topic}",
#     input_vairable = ['topic']
# )
# parser = StrOutputParser()
#
# loading = TextLoader("docoment.txt")
# docs = loading.load()
# print(docs[0].page_content)
# chains = prompt | model | parser
# result = chains.invoke({'topic':docs[0].page_content})
# print(result)
