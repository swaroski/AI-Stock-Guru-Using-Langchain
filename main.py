import streamlit as st
from fpdf import FPDF
import base64
import matplotlib.pyplot as plt
from equity_analyst import equity_analyst

def create_download_link(pdf_data, filename):
    b64 = base64.b64encode(pdf_data)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'



def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)  # To prevent clearing the display

    st.title("AI Financial Analyst App")

    company_name = st.text_input("Company name:")
    analyze_button = st.button("Analyze")

    if analyze_button:
        if company_name:
            try:
                st.write("Analyzing... Please wait.")

                investment_thesis, hist = equity_analyst(company_name)

                # Display current stock price
                current_price = hist['Close'].iloc[-1]
                st.write(f"Current Stock Price: {current_price}")

                st.write("Stock Price Analysis:")

                # Select 'Open' and 'Close' columns from the hist dataframe
                hist_selected = hist[['Open', 'Close']]

                # Create a new figure in matplotlib
                fig, ax = plt.subplots()

                # Plot the selected data
                hist_selected.plot(kind='line', ax=ax)

                # Set the title and labels
                ax.set_title(f"{company_name} Stock Price")
                ax.set_xlabel("Date")
                ax.set_ylabel("Stock Price")

                # Display the plot in Streamlit
                st.pyplot(fig)

                st.write("Investment Thesis / Recommendation:")
                st.markdown(investment_thesis, unsafe_allow_html=True)

                # Generate a PDF report
                report_text = investment_thesis
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font('Arial', 'B', 16)
                pdf.multi_cell(0, 10, report_text)

                # Save the PDF to a byte stream
                pdf_bytes = pdf.output(dest="S").encode("latin-1")

                # Provide the download link for the generated PDF
                html = create_download_link(pdf_bytes, f"{company_name}_investment_thesis")

                st.markdown(html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.write("Please enter the company name.")


if __name__ == "__main__":
    main()
