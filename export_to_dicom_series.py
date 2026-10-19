import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid
import datetime
import os
import zipfile

vol = np.load("sample_dicom/genuine_3d_volume.npy")
nz, ny, nx = vol.shape

dicom_dir = "sample_dicom/clinical_brain_series"
os.makedirs(dicom_dir, exist_ok=True)

study_uid = generate_uid()
series_uid = generate_uid()

print(f"Exporting {nz} slices to genuine clinical DICOM files...")
for z in range(nz):
    slice_data = vol[z, :, :].astype(np.int16)
    filename = os.path.join(dicom_dir, f"CT_brain_slice_{z:03d}.dcm")
    
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2" # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = "CT"
    ds.PatientName = "PATIENT^CLINICAL_HEAD"
    ds.PatientID = "CLARA_BRAIN_3D"
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    
    ds.StudyDate = datetime.date.today().strftime("%Y%m%d")
    ds.SeriesDescription = "Head 3D Brain Routine w/ Contrast"
    ds.InstanceNumber = z + 1
    ds.SliceLocation = float(z * 2.5) # 2.5 mm slice spacing
    ds.ImagePositionPatient = [-128.0, -128.0, float(z * 2.5)]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.SliceThickness = 2.5
    ds.PixelSpacing = [1.0, 1.0]
    
    ds.Rows = ny
    ds.Columns = nx
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1.0
    ds.WindowCenter = 40
    ds.WindowWidth = 80
    
    ds.PixelData = slice_data.tobytes()
    ds.save_as(filename, write_like_original=False)

# Create zip file for web download
zip_path = "sample_dicom/patient_ct_brain_series.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for z in range(nz):
        fname = f"CT_brain_slice_{z:03d}.dcm"
        zipf.write(os.path.join(dicom_dir, fname), arcname=fname)

# Copy slice 38 (tumor peak) as sample_brain_ct_slice.dcm
import shutil
shutil.copy(os.path.join(dicom_dir, "CT_brain_slice_038.dcm"), "sample_dicom/sample_brain_ct_slice.dcm")

print("DICOM packaging complete. ZIP created at", zip_path)
