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
    def test_analyze_route_valid_url(self, mock_analyze_tos, mock_fetch_tos_document):
        mock_fetch_tos_document.return_value = "Sample ToS document"
        mock_analyze_tos.return_value = {
            "categories": [
                {
                    "name": "Sample Category",
                    "weight": 100,
                    "score": 5.0,
                    "explanation": "Sample explanation"
                }
            ],
            "overall_score": 5.0,
            "summary": "Sample summary"
        }
        
        response = self.app.post('/analyze', 
                                 data=json.dumps({'url': 'https://www.example.com/tos'}),
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
