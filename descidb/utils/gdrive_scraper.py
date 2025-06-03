"""
Google Drive scraper for downloading PDFs from public Drive links.
"""

import os
import re
import requests
from typing import List, Dict, Any
import time

from descidb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def extract_drive_folder_id(drive_url: str) -> str:
    """Extract folder ID from Google Drive URL."""
    # Handle different Google Drive URL formats
    patterns = [
        r'/folders/([a-zA-Z0-9-_]+)',
        r'id=([a-zA-Z0-9-_]+)',
        r'/drive/folders/([a-zA-Z0-9-_]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, drive_url)
        if match:
            return match.group(1)
    
    raise ValueError(f"Could not extract folder ID from URL: {drive_url}")


def get_drive_files_list(folder_id: str) -> List[Dict[str, Any]]:
    """Get list of files in a Google Drive folder using web scraping."""
    logger.info("Using web scraping method for public folder access")
    
    # Try to access the folder via web interface to get file IDs
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(folder_url, headers=headers)
        response.raise_for_status()
        
        content = response.text
        
        # Extract file information from the page
        files = []
        
        # Look for file data in various formats
        patterns = [
            r'"([^"]*\.pdf)"[^}]*?"([a-zA-Z0-9_-]{25,})"',
            r'"([a-zA-Z0-9_-]{25,})"[^}]*?"([^"]*\.pdf)"',
            r'\["([^"]*\.pdf)"[^\]]*"([a-zA-Z0-9_-]{25,})"',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if '.pdf' in match[0].lower():
                    file_name, file_id = match
                else:
                    file_id, file_name = match
                
                # Validate file ID (Google Drive IDs are typically 25+ characters)
                if len(file_id) >= 25 and '.pdf' in file_name.lower():
                    files.append({
                        'id': file_id,
                        'name': file_name,
                        'mimeType': 'application/pdf'
                    })
        
        # Remove duplicates
        seen_ids = set()
        unique_files = []
        for file in files:
            if file['id'] not in seen_ids:
                seen_ids.add(file['id'])
                unique_files.append(file)
        
        logger.info(f"Found {len(unique_files)} PDF files via web scraping")
        for file in unique_files:
            logger.info(f"  - {file['name']} (ID: {file['id']})")
        
        return unique_files
        
    except Exception as e:
        logger.error(f"Web scraping failed: {e}")
        return []


def download_pdf_file(file_id: str, file_name: str, output_dir: str) -> str:
    """Download a single PDF file from Google Drive."""
    # Sanitize filename
    safe_filename = re.sub(r'[^\w\-_\.]', '_', file_name)
    if not safe_filename.lower().endswith('.pdf'):
        safe_filename += '.pdf'
    
    output_path = os.path.join(output_dir, safe_filename)
    
    # Skip if file already exists
    if os.path.exists(output_path):
        logger.info(f"File already exists: {safe_filename}")
        return output_path
    
    # Try multiple download methods
    download_methods = [
        f"https://drive.google.com/uc?export=download&id={file_id}",
        f"https://drive.google.com/file/d/{file_id}/view",
        f"https://docs.google.com/uc?export=download&id={file_id}",
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    for download_url in download_methods:
        try:
            logger.info(f"Trying download method: {download_url}")
            
            response = requests.get(download_url, headers=headers, stream=True)
            response.raise_for_status()
            
            # Handle redirects and confirmation pages
            if 'virus scan warning' in response.text.lower() or 'confirm=' in response.url:
                logger.info("Large file detected, extracting confirmation token")
                # Look for confirmation link
                confirm_pattern = r'/uc\?export=download&amp;confirm=([^&]+)&amp;id=' + file_id
                match = re.search(confirm_pattern, response.text)
                if match:
                    confirm_token = match.group(1)
                    download_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                    logger.info(f"Retrying with confirmation token")
                    response = requests.get(download_url, headers=headers, stream=True)
                    response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'application/pdf' in content_type or 'application/octet-stream' in content_type:
                # Save the file
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify the file was downloaded and has content
                if os.path.getsize(output_path) > 1000:  # At least 1KB for a valid PDF
                    logger.info(f"Successfully downloaded: {safe_filename} ({os.path.getsize(output_path)} bytes)")
                    return output_path
                else:
                    logger.warning(f"Downloaded file seems too small: {safe_filename}")
                    os.remove(output_path)
            else:
                logger.warning(f"Unexpected content type: {content_type}")
                
        except Exception as e:
            logger.warning(f"Download method failed: {e}")
            continue
    
    # If all methods failed, raise an error
    raise Exception(f"Failed to download {safe_filename} using all available methods")


def scrape_gdrive_pdfs(drive_url: str) -> List[str]:
    """
    Scrape PDFs from a public Google Drive folder.
    
    Args:
        drive_url: Public Google Drive folder URL
    
    Returns:
        List of paths to downloaded PDF files
    """
    # Always use the papers folder in the root directory
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "papers")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract folder ID from URL
    folder_id = extract_drive_folder_id(drive_url)
    logger.info(f"Extracting PDFs from folder ID: {folder_id}")
    
    # Get list of PDF files in the folder
    pdf_files = get_drive_files_list(folder_id)
    
    if not pdf_files:
        logger.warning("No PDF files found in the Drive folder")
        return []
    
    # Download each PDF file
    downloaded_files = []
    for file_info in pdf_files:
        try:
            file_path = download_pdf_file(
                file_info['id'], 
                file_info['name'], 
                output_dir
            )
            downloaded_files.append(file_path)
            
            # Add small delay to be respectful to Google's servers
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Failed to download {file_info['name']}: {e}")
            continue
    
    logger.info(f"Successfully downloaded {len(downloaded_files)} PDF files")
    return downloaded_files 