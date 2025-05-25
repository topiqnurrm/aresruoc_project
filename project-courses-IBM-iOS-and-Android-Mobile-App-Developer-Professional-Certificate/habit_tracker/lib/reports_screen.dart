import 'dart:convert';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class ReportsScreen extends StatefulWidget {
  const ReportsScreen({super.key});

  @override
  _ReportsScreenState createState() => _ReportsScreenState();
}

class _ReportsScreenState extends State<ReportsScreen> {
  Map<String, List<int>> weeklyData = {};
  List<String> selectedHabits = [];
  final List<String> daysOfWeek = [
    'Mon',
    'Tue',
    'Wed',
    'Thu',
    'Fri',
    'Sat',
    'Sun'
  ];

  @override
  void initState() {
    super.initState();
    _loadWeeklyData();
  }

  Future<void> _loadWeeklyData() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();

    // Get the selected habits from the map
    String? selectedHabitsMapString = prefs.getString('selectedHabitsMap');
    if (selectedHabitsMapString != null) {
      Map<String, dynamic> selectedHabitsMap =
      jsonDecode(selectedHabitsMapString);
      selectedHabits = selectedHabitsMap.keys.toList();
    } else {
      selectedHabits = [];
    }

    // If no habits are selected, reset weeklyData
    if (selectedHabits.isEmpty) {
      setState(() {
        weeklyData = {};
      });
      return;
    }

    // Load the data from shared preferences or generate random mixed data if none exists
    String? storedData = prefs.getString('weeklyData');
    if (storedData == null) {
      Map<String, List<int>> mixedData = {
        for (var habit in selectedHabits)
          habit: List.generate(
              7,
                  (_) =>
              Random().nextBool() ? 1 : 0), // Generate a mix of 0s and 1s
      };
      await prefs.setString('weeklyData', jsonEncode(mixedData));
      storedData = jsonEncode(mixedData);
    }

    // Decode and set weekly data
    setState(() {
      Map<String, dynamic> decodedData = jsonDecode(storedData!);
      weeklyData = decodedData.map((key, value) => MapEntry(
        key,
        List<int>.from(value),
      ));
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.blue.shade700,
        title: const Text(
          'Weekly Report',
          style: TextStyle(
              fontSize: 24, color: Colors.white, fontWeight: FontWeight.bold),
        ),
      ),
      body: weeklyData.isEmpty
          ? const Center(
        child: Text(
          'No data available. Please configure habits first.',
          style: TextStyle(fontSize: 16, color: Colors.grey),
        ),
      )
          : SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: SingleChildScrollView(
          child: DataTable(
            columns: _buildColumns(),
            rows: _buildRows(),
          ),
        ),
      ),
    );
  }

  List<DataColumn> _buildColumns() {
    return [
      const DataColumn(
        label: Text('Habit', style: TextStyle(fontWeight: FontWeight.bold)),
      ),
      ...daysOfWeek.map((day) => DataColumn(
        label: Text(
          day,
          style: const TextStyle(fontWeight: FontWeight.bold),
        ),
      )),
    ];
  }

  List<DataRow> _buildRows() {
    return selectedHabits.map((habit) {
      return DataRow(
        cells: [
          DataCell(Text(habit)),
          ...List.generate(7, (index) {
            bool isCompleted = weeklyData[habit]?[index] == 1;
            return DataCell(
              Icon(
                isCompleted ? Icons.check_circle : Icons.cancel,
                color: isCompleted ? Colors.green : Colors.red,
              ),
            );
          }),
        ],
      );
    }).toList();
  }
}