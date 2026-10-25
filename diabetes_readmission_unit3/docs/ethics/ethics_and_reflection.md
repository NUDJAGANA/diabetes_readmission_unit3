# Ethical Practice and Practical Reflection

## Project purpose and context

This project analyses hospital readmission data and presents findings in a Streamlit dashboard with a prediction simulator. It is an educational project. The model has not been clinically validated and must not be used to make decisions about an individual patient’s care.

The dataset is the UCI *Diabetes 130-US Hospitals for Years 1999–2008* dataset, covering hospital encounters from 130 US hospitals between 1999 and 2008. It is licensed under Creative Commons Attribution 4.0, so the source must be credited. Its age and US hospital setting limit how well its findings represent current healthcare practice or other populations. The licence permits reuse with attribution; it does not by itself establish that every ethical or legal requirement has been met. [UCI dataset description and licence](https://www.archive.ics.uci.edu/dataset/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008)

## Ethical, privacy and governance considerations

### Privacy and data handling

The project data includes fields named `patient_nbr` and `encounter_id`. The encounter table in the app displays encounter identifiers, and the CSV download includes them. Although the project review did not attempt to identify anyone, identifiers and combinations of clinical details can create privacy or linkage risks.

For a public-facing version, encounter identifiers should be removed from displayed tables and downloads unless there is a documented need and appropriate permission to publish them. This is a recommended safeguard; it should not be described as completed unless the app and its downloaded file have actually been changed and checked.

The project review did not verify the dataset’s consent or ethics-review history, or conduct a re-identification assessment. The ICO explains that identifiability can depend on whether information can be linked with other reasonably available data. [ICO guidance on identifiability and anonymisation](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/data-sharing/anonymisation/how-do-we-ensure-anonymisation-is-effective/)

### Legal considerations

This project has not received a legal compliance review. It does not claim compliance with UK GDPR, HIPAA or other data-protection laws. The relevance of a particular law depends on factors such as the data, the organisation and how the data is used. HHS describes two specific methods for HIPAA de-identification; this project has not shown that it meets either method. [HHS guidance on HIPAA de-identification](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html)

Any future use with real patient data or in a healthcare service would need appropriate privacy, legal, security and clinical governance review.

### Fairness and potential harm

The prediction model uses six features supplied by the app’s form. Age, race and gender are not model inputs, but this does not guarantee fair outcomes: other features may still reflect differences between groups.

A subgroup audit was run on 13,998 held-out test records using the validation-selected threshold of 0.52. It compared recall and false-positive rates across age, race and gender groups. The observed false-positive rates ranged from 16.7% to 39.7% across age groups, 26.8% to 32.0% across race groups, and 28.5% to 31.2% by gender. Recall also varied across these groups. Some groups had relatively few actual readmissions, so their estimates were uncertain.

These results are exploratory. They do not prove that the model is biased or unbiased. The audit was not a formal statistical comparison, did not assess every fairness measure or demographic intersection, and did not validate the model clinically or on a separate dataset.

On the held-out test data, overall ROC-AUC was 0.5826, precision was 0.1198, recall was 0.4129 and F1 was 0.1857. These results show important limitations: the model can flag encounters that are not followed by a readmission, and it can miss some readmissions. If a similar tool were used in practice, false alarms could create unnecessary follow-up work or concern, while missed cases could mean missed opportunities for support. Unequal errors could affect groups differently.

The model identifies associations in historical data, such as patterns in prior hospital use; it does not establish that these patterns cause readmission. Treating associations as causes could lead to inaccurate conclusions about patients’ needs or care.

### Responsible design and publication decisions

1. **The prediction form and model were aligned.** A separate model was prepared to accept the six features supplied by the app, and the app was updated to use it. This fixed an input mismatch. It does not establish that the model is accurate or fair.

2. **The app presents the prediction as educational.** It displays a warning that the result should not guide an individual patient’s care. This communicates the model’s intended limits, but it does not replace clinical validation or governance review.

3. **The dashboard explains patterns without claiming causation.** Readmission patterns are presented as associations in historical data, with a note that they do not show that a factor causes readmission.

The app and README should continue to make the dataset source, time period, limitations and intended educational use clear. Encounter identifiers remain a privacy consideration until their removal from public displays and downloads is confirmed.

## Practical reflection

This Unit 3 work extended an existing analysis and dashboard. A practical challenge was that the prediction form and the model it originally used did not accept the same inputs. I prepared a separate six-input model and updated the app to use it. This taught me to check that the form, saved model and prediction code work together.

I checked the simulator using its default inputs. It returned an estimated probability of 59.7%, above the validation-selected threshold of 0.52. This confirmed that the simulator produced an output for that example; it does not demonstrate accuracy for an individual patient. The model’s overall and subgroup results are recorded above.

An external reviewer said the app’s purpose was clear without explanation. They were able to use the age filter, understand the readmission chart and table, use the filters and download the CSV. I also checked the prediction simulator separately; the reviewer did not test that feature.

The model results and subgroup differences showed me why a prediction should be accompanied by clear limitations and why overall performance alone is not enough. If I continue the project, I would investigate subgroup performance further, remove encounter identifiers from public-facing outputs unless their publication is justified, and seek appropriate review before any real-world use.

## Maintenance and future evaluation

The following are future steps, not work completed as part of this review:

- Check that the app, model file, filters, charts and CSV download continue to work after code or dependency changes.
- Review data fields and public outputs before each release, especially identifiers and small-group results.
- If the model or data changes, repeat held-out performance and subgroup evaluations and document the methods, sample sizes and uncertainty.
- If suitable current and representative data becomes available, evaluate whether the model’s performance generalises before making further claims.
- Obtain privacy, legal, security and clinical governance review before considering use with real patient data or in care decisions.