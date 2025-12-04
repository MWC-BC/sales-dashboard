import streamlit as st

# --------------------
# Password Gate
# --------------------
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["access"]["code"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter access code", type="password",
                      on_change=password_entered, key="password")
        return False

    if not st.session_state["password_correct"]:
        st.text_input("Enter access code", type="password",
                      on_change=password_entered, key="password")
        st.error("Wrong password")
        return False

    return True


if not check_password():
    st.stop()

# -----------------------
# Minimal App Under Gate
# -----------------------
st.title("Password Gate Test")
st.write("If you see this after entering the password, everything works.")
