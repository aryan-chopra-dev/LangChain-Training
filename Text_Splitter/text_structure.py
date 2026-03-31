from langchain_text_splitters import RecursiveCharacterTextSplitter

text="""Aim: To identify the key entities, attributes, and relationships for the "MediConnect" system and design
an Entity Relationship Diagram (ERD) representing the structural data requirements.
Problem Statement: "MediConnect" Telemedicine Platform The system requires a robust
database to manage patients, certified doctors, appointment schedules, video consultation sessions,
digital prescriptions, and secure payment transactions. The data model must enforce strict relationships

(e.g., a prescription cannot exist without an appointment) and handle sensitive medical attributes.
1. Identification of Entities and Attributes:   
• Patient: PatientID (PK), Name,MedicalHistory. 
• Doctor: DoctorID (PK), Name, Specialty.       
• Appointment: ApptID (PK), PatientID (FK), DoctorID (FK), Date, Time.
• Prescription: PrescriptionID (PK), ApptID (FK), DrugList.
"""
splitter=RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0
)
chunks=splitter.split_text(text)
print(chunks)