# Error analysis (ft_e3_val on val)

Total errors by type: {"right_label_wrong_evidence": 45, "missing_info_treated_as_proof": 26, "false_insufficient_information": 14, "not_applicable_confusion": 6, "wrong_decision_polarity": 2}

Cases shown: 14 (up to 3 per type, random seed 42)


## Case 1: false_insufficient_information
- annotation_id: 800 | patient: sigir-201515 | type: inclusion
- criterion: Onset of stroke or TIA symptoms within the previous 60 days
- gold label: **included** | predicted: **not enough information**
- gold evidence: [0] Karen is a 72-year-old woman with hypertension and type 2 diabetes, who was hospitalized for cryptogenic stroke two weeks ago.
- predicted evidence: [0] Karen is a 72-year-old woman with hypertension and type 2 diabetes, who was hospitalized for cryptogenic stroke two weeks ago.
- why it matters: (write 1-2 sentences here)


## Case 2: false_insufficient_information
- annotation_id: 22 | patient: sigir-20141 | type: inclusion
- criterion: Clinical diagnosis of one or more risk factors, such as diabetes, smoking habit, lipid metabolism abnormality, history of ischemic heart disease (IHD) or cerebrovascular disease, obesity (BMI>25), chronic heart failure (NYHA II-III), and electrocardiogram (ECG) abnormality (LVH)
- gold label: **included** | predicted: **not enough information**
- gold evidence: [3] She is known to have hypertension and obesity.; [4] She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease.
- predicted evidence: [3] She is known to have hypertension and obesity.; [4] She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease.
- why it matters: (write 1-2 sentences here)


## Case 3: false_insufficient_information
- annotation_id: 21 | patient: sigir-20141 | type: inclusion
- criterion: Clinical diagnosis of hypertension
- gold label: **included** | predicted: **not enough information**
- gold evidence: [3] She is known to have hypertension and obesity.
- predicted evidence: [3] She is known to have hypertension and obesity.
- why it matters: (write 1-2 sentences here)


## Case 4: missing_info_treated_as_proof
- annotation_id: 937 | patient: sigir-201522 | type: inclusion
- criterion: Likely able to produce sputum samples while on study
- gold label: **not enough information** | predicted: **included**
- gold evidence: [0] A 65-year-old male with a history of tuberculosis has started to complain of productive cough with tinges of blood.
- predicted evidence: [4] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- why it matters: (write 1-2 sentences here)


## Case 5: missing_info_treated_as_proof
- annotation_id: 255 | patient: sigir-201415 | type: exclusion
- criterion: 5. Patients with adenomyosis
- gold label: **not enough information** | predicted: **not excluded**
- gold evidence: (none)
- predicted evidence: (none)
- why it matters: (write 1-2 sentences here)


## Case 6: missing_info_treated_as_proof
- annotation_id: 254 | patient: sigir-201415 | type: exclusion
- criterion: 4. Patients with fibroid that is more than 50% sub-mucosal or with hysteroscopically resectable
- gold label: **not enough information** | predicted: **not excluded**
- gold evidence: (none)
- predicted evidence: (none)
- why it matters: (write 1-2 sentences here)


## Case 7: not_applicable_confusion
- annotation_id: 558 | patient: sigir-201430 | type: exclusion
- criterion: Inability to provide informed consent.
- gold label: **not applicable** | predicted: **not excluded**
- gold evidence: [8] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- predicted evidence: [8] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- why it matters: (write 1-2 sentences here)


## Case 8: not_applicable_confusion
- annotation_id: 954 | patient: sigir-201522 | type: exclusion
- criterion: The patient has previously enrolled into this study.
- gold label: **not applicable** | predicted: **not excluded**
- gold evidence: (none)
- predicted evidence: (none)
- why it matters: (write 1-2 sentences here)


## Case 9: not_applicable_confusion
- annotation_id: 24 | patient: sigir-20141 | type: exclusion
- criterion: Patients with IHD within 6 months after percutaneous coronary intervention(PCI), and who are stable but are going to implement PCI or coronary artery bypass grafting(CABG)
- gold label: **not applicable** | predicted: **not excluded**
- gold evidence: (none)
- predicted evidence: (none)
- why it matters: (write 1-2 sentences here)


## Case 10: right_label_wrong_evidence
- annotation_id: 941 | patient: sigir-201522 | type: exclusion
- criterion: Conditions which compromise the subject s ability to take or absorb oral drugs
- gold label: **not excluded** | predicted: **not excluded**
- gold evidence: [4] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- predicted evidence: (none)
- why it matters: (write 1-2 sentences here)


## Case 11: right_label_wrong_evidence
- annotation_id: 804 | patient: sigir-201515 | type: exclusion
- criterion: History of atrial fibrillation
- gold label: **not excluded** | predicted: **not excluded**
- gold evidence: [4] She reports history of occasional palpitations, shortness of breath and chest pain lasting for a few minutes and then stopping on their own.
- predicted evidence: (none)
- why it matters: (write 1-2 sentences here)


## Case 12: right_label_wrong_evidence
- annotation_id: 23 | patient: sigir-20141 | type: exclusion
- criterion: Patients who have already been administered ARB
- gold label: **not excluded** | predicted: **not excluded**
- gold evidence: [5] She currently takes no medications.
- predicted evidence: (none)
- why it matters: (write 1-2 sentences here)


## Case 13: wrong_decision_polarity
- annotation_id: 935 | patient: sigir-201522 | type: inclusion
- criterion: Thought likely to be Mycobacterium culture positive (including NTM infected for the NIH CC) by enrolling physician
- gold label: **not included** | predicted: **included**
- gold evidence: [3] Culture of the sputum revealed an organism with septated, low-angle branching hyphae that had straight, parallel walls.
- predicted evidence: [3] Culture of the sputum revealed an organism with septated, low-angle branching hyphae that had straight, parallel walls.
- why it matters: (write 1-2 sentences here)


## Case 14: wrong_decision_polarity
- annotation_id: 263 | patient: sigir-201415 | type: exclusion
- criterion: 13. Evidence of uterine pathology other than leiomyoma
- gold label: **excluded** | predicted: **not excluded**
- gold evidence: [5] The differential diagnosis includes vesicular mole vs fibroid degeneration.
- predicted evidence: [4] The ultrasound reports enlarged uterus (12 cm x 9 cm x 7 cms) with multiple cystic areas in the interior.
- why it matters: (write 1-2 sentences here)
