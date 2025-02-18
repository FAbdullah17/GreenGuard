import 'package:flutter/material.dart';
import 'home_page.dart';

void main() {
  runApp(const GreenGuardApp());
}

class GreenGuardApp extends StatelessWidget {
  const GreenGuardApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'GreenGuard Mobile',
      theme: ThemeData(
        primarySwatch: Colors.green,
      ),
      home: const HomePage(),
    );
  }
}
