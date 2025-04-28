import subprocess
import os

def generate(name, username, stats):
    print(stats)

    # 1. Create 'PDF' folder if it doesn't exist
    os.makedirs("PDF", exist_ok=True)

    try:
        # 2. Read your Typst template
        with open("template.typ", 'r') as template_file:
            template_content = template_file.read()

        # 3. Fill the template with user's name, username, and emotion stats
        filled_content = template_content.format(
            name,
            username,
            *[f"{x * 100:.2f}" for x in stats]  # Multiply by 100 to convert fractions to percentages
        )

        # 4. Save the filled Typst content into a file
        typ_file_path = f"PDF/{username}.typ"
        with open(typ_file_path, 'w') as report_file:
            report_file.write(filled_content)

        print(f"Generating Typst file for {username}...")

        # 5. Compile the Typst file into a PDF
        pdf_file_path = f"PDF/{username}.pdf"
        result = subprocess.run(
            ["typst", "compile", typ_file_path, pdf_file_path],
            capture_output=True,
            text=True
        )

        # 6. Check if compilation succeeded
        if result.returncode != 0:
            print("❌ Typst compile error:", result.stderr)
        elif not os.path.exists(pdf_file_path):
            print("❌ PDF not created.")
        else:
            print(f"✅ PDF created successfully for {username}.")

    except Exception as e:
        print("❌ Error during report generation:", e)
