import os
import shutil
import yaml
from datetime import datetime

def write_readme_experiment(
    exp_dir: str, 
    title: str = "Experiment Summary", 
    description: str = "", 
    metadata: dict | None = None, 
    yaml_metadata: dict | None = None,
    image_path: str | None = None):
    os.makedirs(exp_dir, exist_ok=True)
    readme_path = os.path.join(exp_dir, "README.md")

    lines = []

    if yaml_metadata:
        yaml_str = yaml.dump(yaml_metadata, sort_keys=False)
        lines.append(f"---\n{yaml_str}---\n\n")

    lines.extend([
        f"# {title}\n",
        f"**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
    ])


    if image_path and os.path.exists(image_path):
        # Create assets directory inside exp_dir
        assets_dir = os.path.join(exp_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)

        # Copy image into assets folder
        image_name = os.path.basename(image_path)
        dest_image_path = os.path.join(assets_dir, image_name)
        shutil.copy(image_path, dest_image_path)

        # Relative path for embedding
        rel_image_path = os.path.relpath(dest_image_path, exp_dir)
        lines.append(f"![Experiment Image]({rel_image_path})\n\n")

    if description:
        lines.append(f"\n## Description\n{description}\n")

    if metadata:
        lines.append("\n## Metadata\n")
        for key, value in metadata.items():
            lines.append(f"- **{key.capitalize()}**: {value}\n")

    with open(readme_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"README.md created at: {readme_path}")
