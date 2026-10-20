import 'package:flutter/material.dart';
import 'screens/landing_page.dart';
import 'theme/app_colors.dart';

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MindSense',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFFCFCFF),
        colorScheme: ColorScheme.fromSeed(seedColor: AppColors.purple),
        fontFamily: 'Roboto',
      ),
      home: const LandingPage(),
    );
  }
}