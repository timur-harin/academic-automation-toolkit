import os
import re
import csv
from bs4 import BeautifulSoup

def extract_text_from_html(file_path):
    """Extract text content from HTML file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean it up
            text = soup.get_text()
            # Remove extra whitespace and newlines
            text = re.sub(r'\s+', ' ', text).strip()
            return text
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def main():
    # Get all folders in current directory that match name surname pattern
    folders = []
    for item in os.listdir('cf'):
        if len(item.split()) >= 2:
            folders.append('./cf/' + item)
    
    results = []
    
    for folder in folders:
        html_file_path = os.path.join(folder, 'onlinetext.html')
        
        if os.path.exists(html_file_path):
            # Extract name and surname from folder name
            name_parts = folder.split()
            name = name_parts[0]
            surname = ' '.join(name_parts[1:]).split('_')[0]  # Handle multiple surnames
        
            text = extract_text_from_html(html_file_path)    
            results.append({
                'First name': name[5:],
                'Last name': surname,
                'CF Handle': text
            })
            #print(f"Processed: {folder}")
        else:
            print(f"File not found: {html_file_path}")
    
    # Write to CSV
    if results:
        csv_filename = 'data/cf_handles.csv'
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['First name', 'Last name', 'CF Handle']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        
        print(f"\nCSV file '{csv_filename}' created successfully!")
        #print(f"Processed {len(results)} folders.")
    else:
        print("No valid folders with HTML files found.")

if __name__ == "__main__":
    main()