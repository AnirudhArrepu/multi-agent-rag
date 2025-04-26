from PyPDF2 import PdfReader

class DocHelper:
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file
    
    def extract_text_from_doc(self):
        if self.uploaded_file.name.endswith('.pdf'):
            reader = PdfReader(self.uploaded_file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            return text
        elif self.uploaded_file.name.endswith('.txt'):
            text = self.uploaded_file.read().decode('utf-8')
            return text

