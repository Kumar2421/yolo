"""
Evaluation script for Helmet Detection model
Author: [Your Name]
"""

from ultralytics import YOLO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import datetime

def generate_pdf_report(metrics, output_dir="evaluation", project_name="Helmet Detection"):
    """Generate a PDF report from evaluation metrics and saved plots."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "evaluation_report.pdf")
    c = canvas.Canvas(report_path, pagesize=letter)

    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, f"{project_name} - Evaluation Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.drawString(50, height - 120, "Metrics:")
    c.drawString(70, height - 140, f"Precision (P): {metrics.box.p.mean():.4f}")
    c.drawString(70, height - 160, f"Recall (R): {metrics.box.r.mean():.4f}")
    c.drawString(70, height - 180, f"mAP@0.5: {metrics.box.map50.mean():.4f}")
    c.drawString(70, height - 200, f"mAP@0.5:0.95: {metrics.box.map.mean():.4f}")

    plot_y = height - 240
    plots = ["confusion_matrix.png", "F1_curve.png", "PR_curve.png", "labels.jpg"]
    for plot in plots:
        path = os.path.join(metrics.save_dir, plot)
        if os.path.exists(path):
            c.drawImage(path, 50, plot_y - 200, width=500, preserveAspectRatio=True, mask='auto')
            plot_y -= 220

    c.save()
    print(f"[INFO] PDF report saved to: {report_path}")

def evaluate_model(model_path, data_yaml_path):
    """Evaluates the model and saves metrics + PDF report."""
    model = YOLO(model_path)

    metrics = model.val(
        data=data_yaml_path,
        split='test',
        plots=True,
        save_json=True
    )

    print("[INFO] Evaluation Metrics:")
    print(f"  Precision (P): {metrics.box.p.mean():.4f}")
    print(f"  Recall (R): {metrics.box.r.mean():.4f}")
    print(f"  mAP@0.5: {metrics.box.map50.mean():.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map.mean():.4f}")

    # Generate PDF report
    generate_pdf_report(metrics)

if __name__ == "__main__":
    model_path = "F:/yolo/yolo/helmet_detection_project/yolov8n_helmet_detection/weights/best.pt"
    data_yaml_path = "F:/yolo/yolo/dataset/data.yaml"
    evaluate_model(model_path, data_yaml_path)
