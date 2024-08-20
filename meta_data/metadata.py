import json


class MetaData:
    def __init__(self):
        self.file_path = "../meta_data/metadata.json"
        with open(self.file_path) as meta_file:
            self.m_data = json.load(meta_file)
        meta_file.close()

    def get_meta_data(self):
        return self.m_data

    def set_meta_data(self, m_data):
        data = json.dumps(m_data, indent=4)
        m_file = open(self.file_path, "w")
        m_file.write(data)
        m_file.close()
