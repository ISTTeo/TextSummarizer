import json

def generate_markdown_from_json(chapters, output_md_file):

    # Generate markdown content
    markdown_content = "# Book Summary\n\n"

    # Generate table of contents
    markdown_content += "## Table of Contents\n\n"
    markdown_content += "- [Summary Over Summaries](#summary-over-summaries)\n"
    markdown_content += "- [Appendix: Detailed Summaries](#appendix-detailed-summaries)\n\n"
    markdown_content += "<div style='page-break-after: always;'></div>\n\n"

    # Generate Summary Over Summaries section
    markdown_content += "# Summary Over Summaries\n\n"
    for i, chapter in enumerate(chapters):
        title = chapter['title']
        summary_over_summary = chapter.get('summary_over_summary', 'No summary over summary available.')
        markdown_content += f"## {title}\n\n"
        markdown_content += f"{summary_over_summary}\n\n"
        markdown_content += f"[View detailed summary](#detailed-summary-{i})\n\n"
        if i < len(chapters) - 1:  # Don't add page break after the last chapter
            markdown_content += "<div style='page-break-after: always;'></div>\n\n"

    # Generate Appendix section
    markdown_content += "# Appendix: Detailed Summaries\n\n"
    for i, chapter in enumerate(chapters):
        title = chapter['title']
        phi_mini_summary = chapter.get('phi_summary', 'No detailed summary available.')
        markdown_content += f"## {title}\n\n"
        markdown_content += f'<a name="detailed-summary-{i}"></a>\n\n'
        markdown_content += f"{phi_mini_summary}\n\n"
        if i < len(chapters) - 1:  # Don't add page break after the last chapter
            markdown_content += "<div style='page-break-after: always;'></div>\n\n"

    # Write the markdown content to a file
    with open(output_md_file, 'w') as md_file:
        md_file.write(markdown_content)

    print(f"Markdown file has been generated: {output_md_file}")

import subprocess

def convert_md_to_pdf(md_file, pdf_file):
    pandoc_command = f"pandoc {md_file} -o {pdf_file} --pdf-engine=xelatex -V mainfont='DejaVu Sans' -V geometry:margin=1in --toc --toc-depth=3 -V colorlinks=true -V linkcolor=blue -V urlcolor=blue"
    
    try:
        subprocess.run(pandoc_command, shell=True, check=True)
        print(f"PDF generated successfully: {pdf_file}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while generating the PDF: {e}")

