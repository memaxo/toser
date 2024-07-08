import unittest
from unittest.mock import patch
from flask import json
from src.app import app
import src.analysis as analysis

class TestToSerApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True 

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'ToSer', response.data)

    def test_analyze_route_no_url(self):
        response = self.app.post('/analyze', 
                                 data=json.dumps({}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'No URL provided')

    def test_analyze_route_invalid_url(self):
        response = self.app.post('/analyze', 
                                 data=json.dumps({'url': 'http://invalidurl.com'}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('Unable to fetch the Terms of Service document', data['error'])

    @patch('src.analysis.fetch_tos_document')
    @patch('src.analysis.analyze_tos')
    @patch('os.environ.get')
    def test_analyze_route_valid_url(self, mock_environ_get, mock_analyze_tos, mock_fetch_tos_document):
        mock_environ_get.return_value = 'fake_api_key'
        mock_fetch_tos_document.return_value = "Google Terms of Service"
        mock_analyze_tos.return_value = {
            "categories": [
                {
                    "name": "Privacy and Data Security",
                    "weight": 25,
                    "score": 6.5,
                    "explanation": "Google's privacy practices are detailed but raise some concerns."
                }
            ],
            "overall_score": 6.5,
            "summary": "Google's Terms of Service are comprehensive but have some areas of concern.",
            "red_flags": ["Extensive data collection practices"]
        }
        
        response = self.app.post('/analyze', 
                                 data=json.dumps({'url': 'https://policies.google.com/terms?hl=en-US'}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('categories', data)
        self.assertIn('overall_score', data)
        self.assertIn('summary', data)

    def test_404_error(self):
        response = self.app.get('/nonexistent-route')
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Not found')

if __name__ == '__main__':
    unittest.main()
