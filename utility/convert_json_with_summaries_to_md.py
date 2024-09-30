import json
from collections import defaultdict

def generate_markdown_from_json(json_file_path, output_md_file):
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        videos = json.load(file)

    # Organize videos by author
    videos_by_author = defaultdict(list)
    for video in videos:
        author_key = video['author'].get('name', 'Unknown Author')
        videos_by_author[author_key].append(video)

    # Generate markdown content
    markdown_content = "# Video Summaries\n\n"

    # Generate table of contents
    markdown_content += "## Table of Contents\n\n"
    markdown_content += "- [Summary Over Summaries](#summary-over-summaries)\n"
    markdown_content += "- [Appendix: Detailed Summaries](#appendix-detailed-summaries)\n\n"
    markdown_content += "<div style='page-break-after: always;'></div>\n\n"

    # Generate Summary Over Summaries section
    markdown_content += "# Summary Over Summaries\n\n"
    for author, author_videos in videos_by_author.items():
        markdown_content += f"## {author}\n\n"
        for i, video in enumerate(author_videos):
            title = video['title']
            summary_over_summary = video.get('summary_over_summary', 'No summary over summary available.')
            markdown_content += f"### {title}\n\n"
            markdown_content += f"{summary_over_summary}\n\n"
            markdown_content += f"[View detailed summary](#detailed-summary-{author.lower().replace(' ', '-')}-{i})\n\n"
        markdown_content += "<div style='page-break-after: always;'></div>\n\n"

    # Generate Appendix section
    markdown_content += "# Appendix: Detailed Summaries\n\n"
    for author, author_videos in videos_by_author.items():
        markdown_content += f"## {author}\n\n"
        for i, video in enumerate(author_videos):
            title = video['title']
            phi_mini_summary = video.get('phi_mini_summary', 'No longer summary available.')
            markdown_content += f"### {title}\n\n"
            markdown_content += f'<a name="detailed-summary-{author.lower().replace(" ", "-")}-{i}"></a>\n\n'
            markdown_content += f"{phi_mini_summary}\n\n"
            markdown_content += "<div style='page-break-after: always;'></div>\n\n"

    # Write the markdown content to a file
    with open(output_md_file, 'w') as md_file:
        md_file.write(markdown_content)

    print(f"Markdown file has been generated: {output_md_file}")

if __name__ == "__main__":
    json_file_path = './videos.json'  # Path to your JSON file
    output_name = "./3.5_video_summaries"
    output_md_file = output_name + ".md"  # Output Markdown file
    generate_markdown_from_json(json_file_path, output_md_file)
