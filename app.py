# Alternative approach using query parameters
import streamlit as st

# ... (keep the rest of the code the same)

# Examples in sidebar
with st.sidebar:
    st.header("📚 Examples")
    examples = {
        "Simple Variable": "x = 5\nprint x",
        "For Loop": "FOR i FROM 1 TO 5\n    PRINT i\nENDFOR", 
        "Conditional": "IF score > 50 THEN\n    PRINT 'Pass'\nELSE\n    PRINT 'Fail'\nENDIF",
        "Array Sum": "numbers = [1, 2, 3, 4, 5]\nsum = 0\nFOR i FROM 0 TO 4\n    sum = sum + numbers[i]\nENDFOR\nprint sum"
    }
    
    # Use query parameters to avoid session state issues
    for name, code in examples.items():
        if st.button(f"📋 {name}", use_container_width=True, key=f"btn_{name}"):
            st.query_params["example"] = name
            st.rerun()

# Check if example was selected via query params
if "example" in st.query_params:
    example_name = st.query_params["example"]
    if example_name in examples:
        st.session_state.pseudocode_input = examples[example_name]
    # Clear the query param
    st.query_params.clear()
