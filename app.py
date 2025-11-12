# app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Page configuration
st.set_page_config(
    page_title="Pseudocode to Python Converter",
    page_icon="🐍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .stButton button:hover {
        background-color: #FF6B6B;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the fine-tuned model from Hugging Face Hub"""
    try:
        # Your uploaded model
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
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate output
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
    
    # Decode output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the Python code part
    if "Python Code:" in full_output:
        python_code = full_output.split("Python Code:")[-1].strip()
    else:
        python_code = full_output[len(prompt):].strip()
    
    return python_code

# Main app
st.markdown('<div class="main-header">🧠 Pseudocode to Python Converter</div>', unsafe_allow_html=True)
st.markdown("Convert your pseudocode into executable Python code using AI!")

# Load model
tokenizer, model = load_model()

if tokenizer and model:
    # Layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input Pseudocode")
        pseudocode = st.text_area(
            "Enter your pseudocode:",
            height=200,
            placeholder="""FOR i FROM 1 TO 5
    PRINT i
ENDFOR""",
            key="pseudocode_input"
        )
    
    with col2:
        st.subheader("🚀 Conversion")
        if st.button("Convert to Python", type="primary", use_container_width=True):
            if pseudocode.strip():
                with st.spinner("🔄 Converting pseudocode..."):
                    try:
                        python_code = pseudocode_to_python(pseudocode, tokenizer, model)
                        
                        st.subheader("✅ Generated Python Code")
                        st.code(python_code, language="python")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Python File",
                            data=python_code,
                            file_name="converted_code.py",
                            mime="text/x-python",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Conversion failed: {e}")
            else:
                st.warning("⚠️ Please enter some pseudocode first!")
    
    # Examples in sidebar
    with st.sidebar:
        st.header("📚 Examples")
        examples = {
            "Simple Variable": "x = 5\nprint x",
            "For Loop": "FOR i FROM 1 TO 5\n    PRINT i\nENDFOR", 
            "Conditional": "IF score > 50 THEN\n    PRINT 'Pass'\nELSE\n    PRINT 'Fail'\nENDIF",
            "Array Sum": "numbers = [1, 2, 3, 4, 5]\nsum = 0\nFOR i FROM 0 TO 4\n    sum = sum + numbers[i]\nENDFOR\nprint sum"
        }
        
        for name, code in examples.items():
            if st.button(f"📋 {name}", use_container_width=True):
                st.session_state.pseudocode_input = code
                st.rerun()

else:
    st.error("❌ Failed to load the model. Please try refreshing the page.")

# Footer
st.markdown("---")
st.markdown("Built with 🐍 Python + Streamlit | Model by [mustehsannisarrao](https://huggingface.co/mustehsannisarrao)")
