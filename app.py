# app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Page configuration
st.set_page_config(
    page_title="Pseudocode to Python Converter",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0F4C81;
        margin-bottom: 2rem;
        text-align: center;
    }
    .example-box {
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 1rem;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .example-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .conversion-box {
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin: 1rem 0;
    }
    .stButton button {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.4);
    }
    .download-btn button {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .code-box {
        background: #2d3748;
        color: #e2e8f0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #FF4B4B;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the fine-tuned model from Hugging Face Hub"""
    try:
        model_name = "mustehsannisarrao/pseudocode-to-python"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def pseudocode_to_python(pseudocode, tokenizer, model):
    """Convert pseudocode to Python code"""
    prompt = f"""Translate the following pseudocode to Python code:

Pseudocode:
{pseudocode}

Python Code:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Python Code:" in full_output:
        python_code = full_output.split("Python Code:")[-1].strip()
    else:
        python_code = full_output[len(prompt):].strip()
    
    return python_code

# Initialize session state
if 'current_pseudocode' not in st.session_state:
    st.session_state.current_pseudocode = ""

# Main app header
st.markdown('<div class="main-header">🧠 Pseudocode to Python Converter</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Transform your pseudocode into executable Python code instantly!</div>', unsafe_allow_html=True)

# Load model
tokenizer, model = load_model()

if tokenizer and model:
    # Define examples
    examples = {
        "🔤 Simple Variable": "x = 5\nprint x",
        "🔄 For Loop": "FOR i FROM 1 TO 5\n    PRINT i\nENDFOR", 
        "⚖️ Conditional": "IF score > 50 THEN\n    PRINT 'Pass'\nELSE\n    PRINT 'Fail'\nENDIF",
        "📊 Array Sum": "numbers = [1, 2, 3, 4, 5]\nsum = 0\nFOR i FROM 0 TO 4\n    sum = sum + numbers[i]\nENDFOR\nprint sum",
        "📋 List Processing": "names = ['Alice', 'Bob', 'Charlie']\nFOR i FROM 0 TO LENGTH(names)-1\n    PRINT 'Hello, ' + names[i]\nENDFOR"
    }
    
    # Main layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Examples section
        st.subheader("🚀 Quick Examples")
        st.write("Click any example to load it:")
        
        # Create example buttons
        for name, code in examples.items():
            if st.button(
                name,
                key=f"example_{name}",
                use_container_width=True,
                help=f"Load: {code[:50]}..."
            ):
                st.session_state.current_pseudocode = code
                st.rerun()
        
        # Input section
        st.subheader("📝 Your Pseudocode")
        pseudocode = st.text_area(
            "Enter or modify your pseudocode below:",
            height=250,
            value=st.session_state.current_pseudocode,
            placeholder="""FOR i FROM 1 TO 5
    PRINT i
ENDFOR""",
            key="pseudocode_input",
            label_visibility="collapsed"
        )
        
        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            convert_clicked = st.button(
                "🎯 Convert to Python", 
                type="primary", 
                use_container_width=True
            )
        with col_btn2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.current_pseudocode = ""
                st.rerun()
    
    with col2:
        st.subheader("✨ Conversion Results")
        
        if convert_clicked and pseudocode.strip():
            with st.spinner("🔄 Converting your pseudocode..."):
                try:
                    python_code = pseudocode_to_python(pseudocode, tokenizer, model)
                    
                    # Success message
                    st.success("✅ Conversion successful!")
                    
                    # Generated code display
                    st.markdown("**Generated Python Code:**")
                    st.markdown(f'<div class="code-box">{python_code}</div>', unsafe_allow_html=True)
                    
                    # Download section
                    st.markdown("---")
                    st.markdown("### 📥 Download Your Code")
                    st.download_button(
                        label="💾 Download as .py file",
                        data=python_code,
                        file_name="converted_code.py",
                        mime="text/x-python",
                        use_container_width=True,
                        key="download_btn"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Conversion failed: {str(e)}")
        
        elif convert_clicked and not pseudocode.strip():
            st.warning("⚠️ Please enter some pseudocode first!")
        
        else:
            # Welcome message when no conversion yet
            st.info("👆 Enter pseudocode on the left and click 'Convert to Python' to see the magic!")
            
            # Features list
            with st.expander("🎉 What you can do:", expanded=True):
                st.markdown("""
                - ✅ Convert pseudocode to Python
                - ✅ Use pre-built examples
                - ✅ Download generated code
                - ✅ Handle loops, conditionals, variables
                - ✅ Get clean, executable Python code
                """)

    # Footer
    st.markdown("---")
    col_foot1, col_foot2, col_foot3 = st.columns([1, 2, 1])
    with col_foot2:
        st.markdown(
            "<div style='text-align: center; color: #666;'>"
            "Built with ❤️ using Streamlit | "
            "Model by <a href='https://huggingface.co/mustehsannisarrao' style='color: #FF4B4B;'>mustehsannisarrao</a>"
            "</div>", 
            unsafe_allow_html=True
        )

else:
    # Error state
    st.error("""
    ❌ Failed to load the model. 
    
    Please try:
    1. Refreshing the page
    2. Checking your internet connection
    3. Ensuring the model is available at: mustehsannisarrao/pseudocode-to-python
    """)

# Add some space at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)
