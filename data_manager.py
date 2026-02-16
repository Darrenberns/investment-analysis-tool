"""
Data Manager for Investment Analysis Tool
Handles persistence, CRUD operations, import/export, and backup/restore
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import shutil
from dcf_engine import Investment, CashFlow


class DataManager:
    """
    Manages persistent storage of investments using JSON files.
    """
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the data manager.
        
        Args:
            data_dir: Directory where data files will be stored
        """
        self.data_dir = Path(data_dir)
        self.data_file = self.data_dir / "investments.json"
        self.backup_dir = self.data_dir / "backups"
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Initialize data file if it doesn't exist
        if not self.data_file.exists():
            self._initialize_data_file()
    
    def _initialize_data_file(self):
        """Create an empty data file with proper structure."""
        initial_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "investments": {}
        }
        self._write_data(initial_data)
    
    def _read_data(self) -> Dict:
        """Read data from JSON file."""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file is corrupted or missing, reinitialize
            self._initialize_data_file()
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    def _write_data(self, data: Dict):
        """Write data to JSON file."""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def save_investment(self, investment: Investment) -> bool:
        """
        Save an investment to persistent storage.
        
        Args:
            investment: Investment object to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = self._read_data()
            
            # Convert investment to dictionary
            inv_dict = investment.to_dict()
            
            # If no ID exists, this is a new investment (shouldn't happen, but safety check)
            if 'id' not in inv_dict or not inv_dict['id']:
                import uuid
                inv_dict['id'] = str(uuid.uuid4())
            
            # Update modified timestamp
            inv_dict['modified_at'] = datetime.now().isoformat()
            
            # Save to data structure
            data['investments'][inv_dict['id']] = inv_dict
            
            # Write to file
            self._write_data(data)
            
            return True
        except Exception as e:
            print(f"Error saving investment: {e}")
            return False
    
    def load_investment(self, investment_id: str) -> Optional[Investment]:
        """
        Load a specific investment by ID.
        
        Args:
            investment_id: Unique identifier for the investment
            
        Returns:
            Investment object or None if not found
        """
        try:
            data = self._read_data()
            
            if investment_id not in data['investments']:
                return None
            
            inv_dict = data['investments'][investment_id]
            return Investment.from_dict(inv_dict)
        except Exception as e:
            print(f"Error loading investment: {e}")
            return None
    
    def load_all_investments(self) -> List[Investment]:
        """
        Load all investments from storage.
        
        Returns:
            List of Investment objects
        """
        try:
            data = self._read_data()
            investments = []
            
            for inv_dict in data['investments'].values():
                try:
                    inv = Investment.from_dict(inv_dict)
                    investments.append(inv)
                except Exception as e:
                    print(f"Error loading investment {inv_dict.get('name', 'unknown')}: {e}")
                    continue
            
            # Sort by created_at (newest first)
            investments.sort(key=lambda x: x.created_at, reverse=True)
            
            return investments
        except Exception as e:
            print(f"Error loading investments: {e}")
            return []
    
    def update_investment(self, investment: Investment) -> bool:
        """
        Update an existing investment.
        
        Args:
            investment: Investment object with updated data
            
        Returns:
            True if successful, False otherwise
        """
        # Update is the same as save for our implementation
        return self.save_investment(investment)
    
    def delete_investment(self, investment_id: str) -> bool:
        """
        Delete an investment from storage.
        
        Args:
            investment_id: Unique identifier for the investment
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = self._read_data()
            
            if investment_id in data['investments']:
                del data['investments'][investment_id]
                self._write_data(data)
                return True
            
            return False
        except Exception as e:
            print(f"Error deleting investment: {e}")
            return False
    
    def list_investment_summaries(self) -> List[Dict]:
        """
        Get a list of investment summaries (without full cash flow data).
        
        Returns:
            List of dictionaries with summary information
        """
        try:
            data = self._read_data()
            summaries = []
            
            for inv_id, inv_dict in data['investments'].items():
                summaries.append({
                    'id': inv_id,
                    'name': inv_dict['name'],
                    'description': inv_dict.get('description', ''),
                    'discount_rate': inv_dict['discount_rate'],
                    'created_at': inv_dict.get('created_at', ''),
                    'modified_at': inv_dict.get('modified_at', ''),
                    'num_cash_flows': len(inv_dict.get('cash_flows', []))
                })
            
            # Sort by created_at (newest first)
            summaries.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            return summaries
        except Exception as e:
            print(f"Error listing investments: {e}")
            return []
    
    def has_persisted_data(self) -> bool:
        """
        Check if there is any persisted data.
        
        Returns:
            True if investments exist in storage
        """
        try:
            data = self._read_data()
            return len(data['investments']) > 0
        except:
            return False
    
    def get_investment_count(self) -> int:
        """
        Get the total number of investments.
        
        Returns:
            Number of investments in storage
        """
        try:
            data = self._read_data()
            return len(data['investments'])
        except:
            return 0
    
    def export_to_file(self, export_path: str) -> bool:
        """
        Export all investments to a JSON file.
        
        Args:
            export_path: Path where the export file should be saved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = self._read_data()
            
            # Add export metadata
            export_data = {
                "export_date": datetime.now().isoformat(),
                "source_version": data.get("version", "1.0"),
                "investments": data["investments"]
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False
    
    def import_from_file(self, import_path: str, overwrite: bool = False) -> tuple[bool, str]:
        """
        Import investments from a JSON file.
        
        Args:
            import_path: Path to the import file
            overwrite: If True, replace existing data; if False, merge
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Read import file
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Validate import file structure
            if 'investments' not in import_data:
                return False, "Invalid import file: missing 'investments' key"
            
            # Read current data
            current_data = self._read_data()
            
            if overwrite:
                # Replace all investments
                current_data['investments'] = import_data['investments']
                message = f"Imported {len(import_data['investments'])} investments (overwrite mode)"
            else:
                # Merge investments
                import_count = 0
                skip_count = 0
                
                for inv_id, inv_dict in import_data['investments'].items():
                    if inv_id in current_data['investments']:
                        skip_count += 1
                    else:
                        current_data['investments'][inv_id] = inv_dict
                        import_count += 1
                
                message = f"Imported {import_count} new investments, skipped {skip_count} existing"
            
            # Save merged data
            self._write_data(current_data)
            
            return True, message
        except Exception as e:
            return False, f"Error importing data: {str(e)}"
    
    def create_backup(self) -> tuple[bool, str]:
        """
        Create a backup of the current data file.
        
        Returns:
            Tuple of (success, backup_path or error_message)
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"investments_backup_{timestamp}.json"
            backup_path = self.backup_dir / backup_filename
            
            # Copy current data file to backup
            shutil.copy2(self.data_file, backup_path)
            
            return True, str(backup_path)
        except Exception as e:
            return False, f"Error creating backup: {str(e)}"
    
    def list_backups(self) -> List[Dict]:
        """
        List all available backups.
        
        Returns:
            List of backup information dictionaries
        """
        try:
            backups = []
            
            for backup_file in self.backup_dir.glob("investments_backup_*.json"):
                backups.append({
                    'filename': backup_file.name,
                    'path': str(backup_file),
                    'size': backup_file.stat().st_size,
                    'created': datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat()
                })
            
            # Sort by creation date (newest first)
            backups.sort(key=lambda x: x['created'], reverse=True)
            
            return backups
        except Exception as e:
            print(f"Error listing backups: {e}")
            return []
    
    def restore_from_backup(self, backup_path: str) -> tuple[bool, str]:
        """
        Restore data from a backup file.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Validate backup file
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            if 'investments' not in backup_data:
                return False, "Invalid backup file format"
            
            # Create a backup of current data before restoring
            self.create_backup()
            
            # Restore the backup
            shutil.copy2(backup_path, self.data_file)
            
            return True, f"Successfully restored from backup"
        except Exception as e:
            return False, f"Error restoring backup: {str(e)}"
    
    def delete_all_investments(self) -> bool:
        """
        Delete all investments (use with caution!).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a backup first
            self.create_backup()
            
            # Reinitialize with empty data
            self._initialize_data_file()
            
            return True
        except Exception as e:
            print(f"Error deleting all investments: {e}")
            return False
    
    def search_investments(self, query: str) -> List[Investment]:
        """
        Search investments by name or description.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching Investment objects
        """
        try:
            all_investments = self.load_all_investments()
            query_lower = query.lower()
            
            matching = [
                inv for inv in all_investments
                if query_lower in inv.name.lower() or 
                   (inv.description and query_lower in inv.description.lower())
            ]
            
            return matching
        except Exception as e:
            print(f"Error searching investments: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about stored investments.
        
        Returns:
            Dictionary with statistics
        """
        try:
            investments = self.load_all_investments()
            
            if not investments:
                return {
                    'total_count': 0,
                    'total_storage_size': 0,
                    'oldest_investment': None,
                    'newest_investment': None
                }
            
            return {
                'total_count': len(investments),
                'total_storage_size': self.data_file.stat().st_size,
                'oldest_investment': min(investments, key=lambda x: x.created_at).name,
                'newest_investment': max(investments, key=lambda x: x.created_at).name,
                'backup_count': len(self.list_backups())
            }
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {'error': str(e)}
