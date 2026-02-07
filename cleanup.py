#!/usr/bin/env python3
"""
Cleanup script for PaperForge AI - removes generated MVPs and Docker artifacts.
Usage: python cleanup.py [--all | --output | --docker]
"""
import subprocess
import shutil
import argparse
from pathlib import Path


def cleanup_output():
    """Remove all generated MVP projects from output directory."""
    output_dir = Path(__file__).parent / "output"

    if not output_dir.exists():
        print("✓ Output directory doesn't exist - nothing to clean")
        return

    count = 0
    for item in output_dir.iterdir():
        if item.is_dir() and item.name != ".gitkeep":
            print(f"  Removing: {item.name}")
            shutil.rmtree(item)
            count += 1
        elif item.is_file() and item.name != ".gitkeep":
            item.unlink()
            count += 1

    print(f"✓ Cleaned {count} items from output/")


def cleanup_docker():
    """Stop and remove PaperForge AI Docker containers and images."""
    # Stop and remove containers
    print("Stopping PaperForge AI containers...")
    result = subprocess.run(
        ["docker", "ps", "-a", "-q", "--filter", "name=paperforge"],
        capture_output=True, text=True
    )
    containers = result.stdout.strip().split()

    for container_id in containers:
        if container_id:
            subprocess.run(["docker", "stop", container_id], capture_output=True)
            subprocess.run(["docker", "rm", container_id], capture_output=True)
            print(f"  Removed container: {container_id[:12]}")

    if not containers or not any(containers):
        print("  No PaperForge AI containers found")

    # Remove images
    print("Removing PaperForge AI images...")
    result = subprocess.run(
        ["docker", "images", "--filter", "reference=paperforge*", "-q"],
        capture_output=True, text=True
    )
    images = result.stdout.strip().split()

    for image_id in images:
        if image_id:
            subprocess.run(["docker", "rmi", "-f", image_id], capture_output=True)
            print(f"  Removed image: {image_id[:12]}")

    if not images or not any(images):
        print("  No PaperForge AI images found")

    print("✓ Docker cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="Clean up PaperForge AI generated files")
    parser.add_argument("--all", action="store_true", help="Clean everything (output + docker)")
    parser.add_argument("--output", action="store_true", help="Clean only output directory")
    parser.add_argument("--docker", action="store_true", help="Clean only Docker containers/images")

    args = parser.parse_args()

    # Default to --all if no flags specified
    if not any([args.all, args.output, args.docker]):
        args.all = True

    print("🧹 PaperForge AI Cleanup\n")

    if args.all or args.output:
        print("Cleaning output directory...")
        cleanup_output()
        print()

    if args.all or args.docker:
        print("Cleaning Docker artifacts...")
        cleanup_docker()
        print()

    print("✅ Cleanup complete!")


if __name__ == "__main__":
    main()
