# Error analysis (ft_e3_test on test)

Total errors by type: {"false_insufficient_information": 15, "right_label_wrong_evidence": 37, "missing_info_treated_as_proof": 28, "not_applicable_confusion": 10}

Cases shown: 12 (up to 3 per type, random seed 42)


## Case 1: false_insufficient_information
- annotation_id: 385 | patient: sigir-201421 | type: inclusion
- criterion: Ability to take and retain oral medication
- gold label: **included** | predicted: **not enough information**
- gold evidence: [4] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- predicted evidence: (none)
- why it matters: (write 1-2 sentences here)


## Case 2: false_insufficient_information
- annotation_id: 162 | patient: sigir-201410 | type: inclusion
- criterion: Patients undergoing open abdominal aortic aneurysm repair
- gold label: **not included** | predicted: **not enough information**
- gold evidence: (none)
- predicted evidence: (none)
- why it matters: (write 1-2 sentences here)


## Case 3: false_insufficient_information
- annotation_id: 161 | patient: sigir-201410 | type: inclusion
- criterion: Patients undergoing elective carotid endarterectomy
- gold label: **not included** | predicted: **not enough information**
- gold evidence: (none)
- predicted evidence: (none)
- why it matters: (write 1-2 sentences here)


## Case 4: missing_info_treated_as_proof
- annotation_id: 418 | patient: sigir-201422 | type: exclusion
- criterion: Failure to pass MRI metal screening
- gold label: **not enough information** | predicted: **not excluded**
- gold evidence: (none)
- predicted evidence: (none)
- why it matters: (write 1-2 sentences here)


## Case 5: missing_info_treated_as_proof
- annotation_id: 393 | patient: sigir-201421 | type: exclusion
- criterion: Treatment with anti-rheumatic/immunosuppressive drugs within 3 months prior to first dose of study medication, other than the following medications at stable doses: methotrexate (≤25 mg/week), azathioprine (≤2.5 mg/kg/day), hydroxychloroquine and mycophenolate mofetil (≤3000 mg/day).
- gold label: **not enough information** | predicted: **not excluded**
- gold evidence: (none)
- predicted evidence: (none)
- why it matters: (write 1-2 sentences here)


## Case 6: missing_info_treated_as_proof
- annotation_id: 392 | patient: sigir-201421 | type: exclusion
- criterion: Intravenous cyclophosphamide within 6 months prior to the first dose of study medication.
- gold label: **not enough information** | predicted: **not excluded**
- gold evidence: (none)
- predicted evidence: (none)
- why it matters: (write 1-2 sentences here)


## Case 7: not_applicable_confusion
- annotation_id: 697 | patient: sigir-20158 | type: exclusion
- criterion: Pregnant
- gold label: **not applicable** | predicted: **not excluded**
- gold evidence: (none)
- predicted evidence: (none)
- why it matters: (write 1-2 sentences here)


## Case 8: not_applicable_confusion
- annotation_id: 693 | patient: sigir-20158 | type: exclusion
- criterion: Suspected diagnosis of a sleep disorder other than OSA (i.e., periodic leg movements, narcolepsy, insomnia, central sleep apnea, sleep hypoventilation syndrome)
- gold label: **not applicable** | predicted: **not excluded**
- gold evidence: (none)
- predicted evidence: (none)
- why it matters: (write 1-2 sentences here)


## Case 9: not_applicable_confusion
- annotation_id: 683 | patient: sigir-20158 | type: exclusion
- criterion: Previous tonsillectomy, including partial tonsillectomy
- gold label: **not applicable** | predicted: **not excluded**
- gold evidence: (none)
- predicted evidence: (none)
- why it matters: (write 1-2 sentences here)


## Case 10: right_label_wrong_evidence
- annotation_id: 840 | patient: sigir-201516 | type: inclusion
- criterion: Negative allergy skin test as performed in the screening protocol.
- gold label: **not enough information** | predicted: **not enough information**
- gold evidence: (none)
- predicted evidence: [1] He has had a history of allergic rhinitis, but no history of wheezing.
- why it matters: (write 1-2 sentences here)


## Case 11: right_label_wrong_evidence
- annotation_id: 307 | patient: sigir-201417 | type: inclusion
- criterion: Patient selection should be cautious, covering basically 5 types of liver damage which must be submitted prior to liver volumetry: Metastatic Disease subsidiary right hepatectomy extended to segment IV Metastatic Disease subsidiary right hepatectomy with suspected diseased liver (neoadjuvant chemotherapy) (in cases of doubt may be used liver function test indocyanine green) Bilobar liver metastases with multiple nodules in the right lobe and more than 3 nodules greater than 30 mm in the left hepatic lobe (LHI) will perform lumpectomies the LHI + right portal branch ligation (or postoperative percutaneous embolization) in order to make right hepatectomy 4-6 weeks (two stage surgery) Subsidiary Hepatocarcinoma extended right hepatectomy Liver Injury benign / malignant (Hemangiomas, hydatid cysts or liver tumors / primary bile hepatoblastoma), which by extension threatens the viability of the remaining liver tissue.
- gold label: **not enough information** | predicted: **not enough information**
- gold evidence: [3] Exploratory laparotomy reveals a ruptured liver abscess, which is then surgically drained.
- predicted evidence: [3] Exploratory laparotomy reveals a ruptured liver abscess, which is then surgically drained.; [4] After surgery, the patient is taken to the ICU.
- why it matters: (write 1-2 sentences here)


## Case 12: right_label_wrong_evidence
- annotation_id: 769 | patient: sigir-201513 | type: exclusion
- criterion: Infection of the deep tissues of the upper respiratory tract (eg, epiglottitis, retropharyngeal or buccal cellulitis, or abscess of the retropharynx, tonsil, or peritonsillar area) or of the suprapharyngeal respiratory tract and its connecting structures (eg, sinusitis, otitis media, or orbital/periorbital cellulitis)
- gold label: **not excluded** | predicted: **not excluded**
- gold evidence: (none)
- predicted evidence: [0] A 5-year-old boy presents to the emergency department with complaints of progressively worsening dysphagia, drooling, fever and vocal changes.; [1] He is toxic-appearing, and leans forward while sitting on his mother's lap.; [2] He is drooling and speaks with a muffled "hot potato" voice.
- why it matters: (write 1-2 sentences here)
