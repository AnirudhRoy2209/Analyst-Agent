import os
import requests
import pandas as pd

class etltools:

    def __init__(self):
        pass

    def extract_load(self, url:str, output_folder:str, format:str):
        """
        This tool extracts the data from the API (url) and loads it into the
        the desired location (output_folder).

        Args:
            url (str): The API endpoint from which to extract data.
            output_folder (str): The folder where the extracted data will be saved.
        
        Returns:
            str: A message indicating the success or failure of the operation.

        """
        if not url or not url.startswith(("http://", "https://")):
            return "I need a valid API URL to extract from. Please provide one."

        project_root=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        output_folder=os.path.join(project_root, output_folder)

        try:
            response=requests.get(url)
            response.raise_for_status()
            data=response.json()

            filename= os.path.join(output_folder, f"extracted data.{format}")
            os.makedirs(output_folder, exist_ok=True)

            if isinstance(data, dict) and "results" in data:
                df = pd.json_normalize(data["results"])
            elif isinstance(data, list):
                df = pd.json_normalize(data)
            else:
                df = pd.json_normalize([data])   
            
            if format=="csv":
                df.to_csv(filename, index=False)
            elif format=="json":
                df.to_json(filename, orient="records", lines=True)
            elif format=="parquet":
                df.to_parquet(filename, index=False)
            else:
                return f"Unsupported format: {format}"

            return f"data successfully extracted and saved to {filename}"
        except requests.exceptions.RequestException as e:
            return f"failed to extract data: {e}"



    def transform_load_context(self, file_path:str):
        """
        This tool transforms the data from the specified file and loads it into the
        desired location (output_folder).

        Args:
            file_path (str): The path to the file containing the data to be transformed.
            output_folder (str): The folder where the transformed data will be saved.
            output_format (str): The format in which to save the transformed data (csv, json, parquet).
        Returns:
            str: A message indicating the success or failure of the operation.
        """
        if not file_path or not os.path.exists(file_path):
            return f"File not found: {file_path}"

        file_extension=os.path.splitext(file_path)[1].lower()
        if file_extension ==".csv":
            df= pd.read_csv(file_path)
        elif file_extension ==".json":
            df=pd.read_json(file_path)
        elif file_extension ==".parquet":
            df=pd.read_parquet(file_path)
        else:
            return f"unsupported file format: {file_extension}"

        top_3_rows=str(df.head(3))

        return top_3_rows

    def execute_code(self, code:str):
        """
        This tool executes the provided code and returns the output.

        Args:
            code (str): The code to be executed.
        Returns:
            str: The output of the executed code or an error message if execution fails.
        """
        try:
            exec(code)
            return "code executed successfully"
        except Exception as e:
            return f"failed to execute code: {e}"
        





if __name__=="__main__":
    obj= etltools()
    path= "C:\\Users\\ANIDUDH\\OneDrive\\Desktop\\projects\\Analyst Agent\\Analyst Agent\\data\\extract\\extracted data.csv"
    print(obj.transform_load_context(path))