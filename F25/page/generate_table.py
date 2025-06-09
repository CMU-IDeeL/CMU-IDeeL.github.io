# generate_table.py
import yaml
from jinja2 import Environment, FileSystemLoader

def generate_html_table(filename, template_filename, output_filename):
    # Load data from YAML file
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)

    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader('.')) # Looks for templates in the current directory
    template = env.get_template(template_filename)

    # Render the template with data
    html_output = template.render(data=data)

    # Save the output to an HTML file
    with open(output_filename, 'w') as f:
        f.write(html_output)

    print(f"HTML table generated successfully as {output_filename}!")

html_tables = [
    ["tables_data/recitations.yaml", "tables_templates/recitations_template.html", "tables/recitations.html"],
    ["tables_data/lectures.yaml", "tables_templates/lectures_table_template.html", "tables/lectures_table.html"],
    ["tables_data/assignments.yaml", "tables_templates/assignments_template.html", "tables/assignments_table.html"]
]

for filename, template_filename, output_filename in html_tables:
    generate_html_table(filename, template_filename, output_filename)