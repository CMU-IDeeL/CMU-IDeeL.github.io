# generate_table.py
import os
import yaml
from jinja2 import Environment, FileSystemLoader

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAGES_DIR = os.path.dirname(SCRIPT_DIR)  # Parent directory (pages/)

def generate_html_table(filename, template_filename, output_filename):
    # Load data from YAML file
    yaml_path = os.path.join(SCRIPT_DIR, filename)
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Set up Jinja2 environment - use pages/ as base so we can access tables_templates/
    env = Environment(loader=FileSystemLoader(PAGES_DIR))
    template = env.get_template(template_filename)

    # Render the template with data
    html_output = template.render(data=data)

    # Save the output to an HTML file
    output_path = os.path.join(PAGES_DIR, output_filename)
    with open(output_path, 'w') as f:
        f.write(html_output)

    print(f"HTML table generated successfully as {output_filename}!")

html_tables = [
    ["recitations.yaml", "tables_templates/recitations_template.html", "tables/recitations.html"],
    ["lectures.yaml", "tables_templates/lectures_table_template.html", "tables/lectures_table.html"],
    ["assignments.yaml", "tables_templates/assignments_template.html", "tables/assignments_table.html"]
] # relative path to the tables_data folder. DO NOT use "../" for other relative filepaths here, apparently Jinja2 doesnt like ../

for filename, template_filename, output_filename in html_tables:
    generate_html_table(filename, template_filename, output_filename)